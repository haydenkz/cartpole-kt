package com.example.cartpolekt.simulation

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.cartpolekt.ml.MemoryItem
import com.example.cartpolekt.ml.PPOAgent
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class CartPoleViewModel : ViewModel() {
    private val model = CartPoleModel()
    private val agent = PPOAgent(inputSize = 5, hiddenSize = 64, actionSize = 2)

    private val _state = MutableStateFlow(model.reset())
    val state: StateFlow<CartPoleState> = _state.asStateFlow()

    private val _mode = MutableStateFlow(0)
    val mode: StateFlow<Int> = _mode.asStateFlow()

    private val _episodeCount = MutableStateFlow(0)
    val episodeCount: StateFlow<Int> = _episodeCount.asStateFlow()

    private val _rewardHistory = MutableStateFlow<List<Float>>(emptyList())
    val rewardHistory: StateFlow<List<Float>> = _rewardHistory.asStateFlow()

    private var simulationJob: Job? = null

    private var isLeftPressed = false
    private var isRightPressed = false

    private var currentEpisodeReward = 0f
    private var stepsInEpisode = 0
    private val maxSteps = 700

    init {
        startLoop()
    }

    fun setMode(newMode: Int) {
        _mode.value = newMode
        reset()
    }

    private fun startLoop() {
        simulationJob?.cancel()
        simulationJob = viewModelScope.launch {
            while (isActive) {
                when (_mode.value) {
                    0 -> {
                        manualStep()
                        delay(20)
                    }
                    1 -> {
                        repeat(20) {
                            trainStep()
                        }
                        delay(1)
                    }
                    2 -> {
                        evalStep()
                        delay(20)
                    }
                }
            }
        }
    }

    private fun manualStep() {
        val action = when {
            isLeftPressed && !isRightPressed -> 0
            !isLeftPressed && isRightPressed -> 1
            else -> -1
        }

        _state.value = model.step(_state.value, action)

        if (abs(_state.value.x) > model.xThreshold || abs(_state.value.theta) > model.thetaThresholdRadians) {
            // For manual mode we simply let the user continue; optional reset could go here.
        }
    }

    private fun trainStep() {
        val currentState = _state.value

        val stateVec = buildStateVector(currentState)

        val (action, prob, value) = agent.getAction(stateVec)
        val nextState = model.step(currentState, action)

        val uprightAlignment = cos(nextState.theta) // -1 bottom, +1 upright
        val posFrac = (abs(nextState.x) / model.xThreshold).coerceIn(0f, 1f)

        val uprightTerm = 2.0f * uprightAlignment
        val centerPenalty = 2.0f * posFrac * posFrac
        val edgeProximity = (posFrac - 0.8f).coerceIn(0f, 1f)
        val wallHugPenalty = 6.0f * edgeProximity * edgeProximity * edgeProximity // harsh near edges
        val centeredBalanceBonus = if (abs(nextState.theta) < 0.12f) {
            // Reward being upright while keeping the cart near center; ramps up as x approaches 0
            4.0f * (1f - posFrac)
        } else 0f
        val velPenalty = 0.15f * (nextState.xDot / 3f) * (nextState.xDot / 3f)
        val angVelPenalty = 0.15f * (nextState.thetaDot / 6f) * (nextState.thetaDot / 6f)
        val balanceBonus = if (abs(nextState.theta) < 0.08f && abs(nextState.x) < 0.2f && abs(nextState.thetaDot) < 0.3f && abs(nextState.xDot) < 0.3f) 3.0f else 0f
        val quickBalanceBonus = if (balanceBonus > 0f) {
            // Earlier stable balance gets extra reward to encourage quick recovery
            val speedFactor = ((maxSteps - stepsInEpisode).toFloat() / maxSteps.toFloat()).coerceIn(0f, 1f)
            5.0f * speedFactor
        } else 0f

        var reward = uprightTerm -
            centerPenalty -
            wallHugPenalty -
            velPenalty -
            angVelPenalty +
            centeredBalanceBonus +
            balanceBonus +
            quickBalanceBonus

        val nextStepCount = stepsInEpisode + 1
        val tippedOver = abs(nextState.theta) > model.thetaThresholdRadians
        val done = tippedOver || nextStepCount >= maxSteps

        val nextStateVec = buildStateVector(nextState)

        if (tippedOver) reward -= 3.0f
        reward = reward.coerceIn(-10f, 10f)

        agent.store(MemoryItem(stateVec, action, reward, nextStateVec, done, prob, value))

        currentEpisodeReward += reward
        stepsInEpisode = nextStepCount
        _state.value = nextState

        if (done) {
            _episodeCount.value += 1
            val currentHistory = _rewardHistory.value.toMutableList()
            currentHistory.add(currentEpisodeReward)
            _rewardHistory.value = currentHistory

            if (_episodeCount.value % 10 == 0) {
                Log.d("CartPole", "Episode ${_episodeCount.value} finished. Reward: $currentEpisodeReward Steps: $stepsInEpisode")
            }

            _state.value = model.reset()
            currentEpisodeReward = 0f
            stepsInEpisode = 0
        }
    }

    private fun evalStep() {
        val currentState = _state.value

        val stateVec = buildStateVector(currentState)

        val probs = agent.actor.forward(stateVec)
        var action = if (probs[0] > probs[1]) 0 else 1

        if (isLeftPressed && !isRightPressed) {
            action = 0
        } else if (!isLeftPressed && isRightPressed) {
            action = 1
        }

        val nextState = model.step(currentState, action)
        // In eval, reset only if angle exceeds threshold or after max steps.
        val done = abs(nextState.theta) > model.thetaThresholdRadians

        _state.value = nextState

        if (done) {
            _state.value = model.reset(startUpright = false)
        }
    }

    fun setLeftPressed(pressed: Boolean) {
        isLeftPressed = pressed
    }

    fun setRightPressed(pressed: Boolean) {
        isRightPressed = pressed
    }

    fun reset() {
        _state.value = if (_mode.value == 2) {
            model.reset(startUpright = false)
        } else {
            model.reset(startUpright = false)
        }
        isLeftPressed = false
        isRightPressed = false
        if (_mode.value == 1) {
            agent.memory.clear()
            currentEpisodeReward = 0f
            stepsInEpisode = 0
        }
    }

    private fun buildStateVector(state: CartPoleState): FloatArray {
        val normX = state.x / model.xThreshold
        val normXDot = state.xDot / 2.0f
        val sinTheta = sin(state.theta)
        val cosTheta = cos(state.theta)
        val normThetaDot = state.thetaDot / 4.0f
        return floatArrayOf(normX, normXDot, sinTheta, cosTheta, normThetaDot)
    }
}
