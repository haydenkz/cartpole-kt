package com.example.cartpolekt.ml

import kotlin.math.max
import kotlin.math.min
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

data class MemoryItem(
    val state: Vector,
    val action: Int,
    val reward: Float,
    val nextState: Vector,
    val done: Boolean,
    val prob: Float, // Probability of the action taken
    val value: Float // Critic value
)

class PPOAgent(val inputSize: Int, val hiddenSize: Int, val actionSize: Int) {
    
    val actor = SimpleNetwork(inputSize, hiddenSize, actionSize, "softmax")
    val critic = SimpleNetwork(inputSize, hiddenSize, 1, "linear")
    
    val actorOpt = AdamOptimizer(lr = 0.003f)
    val criticOpt = AdamOptimizer(lr = 0.0045f)

    val memory = mutableListOf<MemoryItem>()
    
    val gamma = 0.99f
    val lambda = 0.95f
    val epsClip = 0.2f
    val kEpochs = 4
    private val valueCoef = 0.5f
    var entropyCoef = 0.015f
        private set
    val batchSize = 160

    fun getAction(state: Vector): Triple<Int, Float, Float> {
        val probs = actor.forward(state)
        // Value estimate
        val v = critic.forward(state)[0]
        
        // Sampling
        val r = kotlin.random.Random.nextFloat()
        var cum = 0f
        var action = probs.lastIndex
        var prob = probs.last()
        
        for (i in probs.indices) {
            cum += probs[i]
            if (r < cum) {
                action = i
                prob = probs[i]
                break
            }
        }
        return Triple(action, prob, v)
    }

    fun store(item: MemoryItem) {
        memory.add(item)
        
        // Check if we have enough data to update
        if (memory.size >= batchSize) {
            update()
        }
    }
    
    private fun update() {
        if (memory.isEmpty()) return

        // --- Generalized Advantage Estimation (GAE) ---
        val rewards = FloatArray(memory.size)
        val advantages = FloatArray(memory.size)
        
        // We need V(next) for the last step. 
        // Since we update mid-episode or end-episode, if it's not 'done', we theoretically need 
        // value of next state. For simplicity in this fixed batch logic, 
        // we'll assume 0 value for next state of the last sample in batch if unknown,
        // OR we should actually compute it.
        // But `store` just pushed the last item. We have `nextState` in MemoryItem.
        // Let's compute last next value.
        val lastItem = memory.last()
        val lastNextVal = if (lastItem.done) 0f else critic.forward(lastItem.nextState)[0]
        
        var lastGaeLam = 0f
        
        for (i in memory.indices.reversed()) {
            val item = memory[i]
            val nextVal = if (i == memory.lastIndex) lastNextVal else memory[i+1].value
            val mask = if (item.done) 0f else 1f
            
            val delta = item.reward + gamma * nextVal * mask - item.value
            lastGaeLam = delta + gamma * lambda * mask * lastGaeLam
            advantages[i] = lastGaeLam
            
            // Return = Advantage + Value
            rewards[i] = advantages[i] + item.value
        }

        // Normalize Advantages
        val advMean = advantages.average().toFloat()
        val advStd = sqrt(advantages.map { val d = it - advMean; d * d }.average()).toFloat() + 1e-8f
        for (i in advantages.indices) {
            advantages[i] = (advantages[i] - advMean) / advStd
        }

        // 2. Training Loop
        for (k in 0 until kEpochs) {
            for (i in memory.indices) {
                val item = memory[i]
                val state = item.state
                val action = item.action
                val oldProb = item.prob
                val targetValue = rewards[i]
                val advantage = advantages[i]

                // --- CRITIC UPDATE ---
                // Value V(s)
                val cL1Z = critic.l1.weights.matmul(state).add(critic.l1.biases)
                val cL1A = FloatArray(cL1Z.size) { if (cL1Z[it] > 0) cL1Z[it] else 0f } // ReLU
                val cOutZ = critic.l2.weights.matmul(cL1A).add(critic.l2.biases)
                val value = cOutZ[0] 
                
                // MSE Loss: (V(s) - Return)^2
                val criticDelta = valueCoef * 2 * (value - targetValue)
                
                // Backprop Critic L2
                for (j in 0 until critic.l2.inputSize) {
                    critic.l2.weightGradients[0][j] += criticDelta * cL1A[j]
                }
                critic.l2.biasGradients[0] += criticDelta
                
                // Backprop Critic L1
                val dL1 = FloatArray(critic.l1.outputSize)
                for (j in 0 until critic.l1.outputSize) {
                    dL1[j] = criticDelta * critic.l2.weights[0][j] * (if (cL1Z[j] > 0) 1f else 0f) 
                }
                
                for (r in 0 until critic.l1.outputSize) {
                    for (c in 0 until critic.l1.inputSize) {
                        critic.l1.weightGradients[r][c] += dL1[r] * state[c]
                    }
                    critic.l1.biasGradients[r] += dL1[r]
                }
                

                // --- ACTOR UPDATE ---
                val aL1Z = actor.l1.weights.matmul(state).add(actor.l1.biases)
                val aL1A = FloatArray(aL1Z.size) { if (aL1Z[it] > 0) aL1Z[it] else 0f }
                val aOutZ = actor.l2.weights.matmul(aL1A).add(actor.l2.biases)
                
                // Softmax
                val maxZ = aOutZ.maxOrNull() ?: 0f
                val exps = aOutZ.map { v: Float -> exp((v - maxZ).toDouble()).toFloat() }
                val sumExps = exps.sum()
                val probs = exps.map { v: Float -> v / sumExps }
                val newProb = probs[action]

                // Ratio
                val ratio = newProb / oldProb
                
                // PPO Loss (Maximize Objective)
                // Clip only when the ratio moves in a direction that would
                // improve the objective, otherwise keep the real gradient.
                val shouldClip = (advantage > 0f && ratio > 1 + epsClip) ||
                    (advantage < 0f && ratio < 1 - epsClip)
                val dObj_dProb = if (shouldClip) 0f else advantage / oldProb
                
                // Entropy
                val H = -probs.map { v: Float -> if (v > 1e-7f) v * ln(v) else 0f }.sum()
                
                val dObj_dZ = FloatArray(actor.l2.outputSize)
                for (j in dObj_dZ.indices) {
                    val delta = if (j == action) 1f else 0f
                    
                    // PG Term
                    val pgTerm = dObj_dProb * (newProb * (delta - probs[j]))
                    
                    // Entropy Term
                    val p = probs[j]
                    val logP = if (p > 1e-7f) ln(p) else 0f
                    val entropyTerm = -p * (logP + H)
                    
                    dObj_dZ[j] = pgTerm + entropyCoef * entropyTerm
                }
                
                // Minimize Negative Objective -> Gradient is -dObj/dTheta
                val dLoss_dZ = dObj_dZ.map { -it }.toFloatArray()
                
                // Backprop Actor L2
                for (r in 0 until actor.l2.outputSize) {
                    for (c in 0 until actor.l2.inputSize) {
                        actor.l2.weightGradients[r][c] += dLoss_dZ[r] * aL1A[c]
                    }
                    actor.l2.biasGradients[r] += dLoss_dZ[r]
                }

                // Backprop Actor L1
                val dL1Actor = FloatArray(actor.l1.outputSize)
                for (j in 0 until actor.l1.outputSize) {
                    var sum = 0f
                    for (k in 0 until actor.l2.outputSize) {
                        sum += dLoss_dZ[k] * actor.l2.weights[k][j]
                    }
                    dL1Actor[j] = sum * (if (aL1Z[j] > 0) 1f else 0f)
                }

                for (r in 0 until actor.l1.outputSize) {
                    for (c in 0 until actor.l1.inputSize) {
                        actor.l1.weightGradients[r][c] += dL1Actor[r] * state[c]
                    }
                    actor.l1.biasGradients[r] += dL1Actor[r]
                }
            }
            
            val scale = 1.0f / memory.size
            normalizeGradients(actor, scale)
            normalizeGradients(critic, scale)
            
            actorOpt.step(actor.getParams())
            criticOpt.step(critic.getParams())
        }
        
        memory.clear()
    }
    
    fun normalizeGradients(net: SimpleNetwork, scale: Float) {
        for (layer in net.getParams()) {
            for (i in layer.weightGradients.indices) {
                for (j in layer.weightGradients[i].indices) {
                    layer.weightGradients[i][j] *= scale
                }
            }
            for (i in layer.biasGradients.indices) {
                layer.biasGradients[i] *= scale
            }
        }
    }
    fun setEntropyCoefficient(weight: Float) {
        entropyCoef = max(0f, weight)
    }
}
