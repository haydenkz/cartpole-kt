package com.example.cartpolekt.ml

import kotlin.math.exp
import kotlin.math.max
import kotlin.random.Random

// Minimal Vector/Matrix helpers
typealias Vector = FloatArray
typealias Matrix = Array<FloatArray>

fun Vector.dot(other: Vector): Float {
    var sum = 0f
    for (i in this.indices) sum += this[i] * other[i]
    return sum
}

fun Matrix.matmul(vec: Vector): Vector {
    val result = FloatArray(this.size)
    for (i in this.indices) {
        result[i] = this[i].dot(vec)
    }
    return result
}

fun Vector.add(other: Vector): Vector {
    val res = FloatArray(this.size)
    for (i in this.indices) res[i] = this[i] + other[i]
    return res
}

fun Vector.sub(other: Vector): Vector {
    val res = FloatArray(this.size)
    for (i in this.indices) res[i] = this[i] - other[i]
    return res
}

fun Vector.scale(scalar: Float): Vector {
    val res = FloatArray(this.size)
    for (i in this.indices) res[i] = this[i] * scalar
    return res
}

// Simple Dense Layer with He Initialization
class DenseLayer(val inputSize: Int, val outputSize: Int, val activation: String = "relu") {
    // Weights: [outputSize][inputSize]
    val weights: Matrix = Array(outputSize) { FloatArray(inputSize) }
    val biases: Vector = FloatArray(outputSize)
    
    // Gradients
    var weightGradients: Matrix = Array(outputSize) { FloatArray(inputSize) }
    var biasGradients: Vector = FloatArray(outputSize)

    // Adam optimizer state
    var mW: Matrix = Array(outputSize) { FloatArray(inputSize) }
    var vW: Matrix = Array(outputSize) { FloatArray(inputSize) }
    var mB: Vector = FloatArray(outputSize)
    var vB: Vector = FloatArray(outputSize)
    var t: Int = 0

    init {
        val limit = kotlin.math.sqrt(6.0 / (inputSize + outputSize)).toFloat()
        for (i in 0 until outputSize) {
            for (j in 0 until inputSize) {
                weights[i][j] = Random.nextFloat() * 2 * limit - limit
            }
            biases[i] = 0f
        }
    }

    fun forward(input: Vector): Vector {
        val z = weights.matmul(input).add(biases)
        return when (activation) {
            "relu" -> FloatArray(z.size) { if (z[it] > 0) z[it] else 0f }
            "softmax" -> {
                val maxVal = z.maxOrNull() ?: 0f
                val exps = z.map { exp(it - maxVal) }
                val sum = exps.sum()
                FloatArray(z.size) { exps[it] / sum }
            }
            "tanh" -> FloatArray(z.size) { kotlin.math.tanh(z[it]) }
            else -> z // linear
        }
    }
    
    // Helper for backprop: derivative of activation function
    // z is the pre-activation value. For ReLU we often use the output value if we know it was positive.
    // We will simplify backprop by passing inputs and outputs during the backward pass logic in the network.
}

class SimpleNetwork(
    val inputSize: Int,
    val hiddenSize: Int,
    val outputSize: Int,
    val outputActivation: String
) {
    val l1 = DenseLayer(inputSize, hiddenSize, "relu")
    val l2 = DenseLayer(hiddenSize, outputSize, outputActivation)

    fun forward(input: Vector): Vector {
        val h = l1.forward(input)
        return l2.forward(h)
    }

    fun getParams(): List<DenseLayer> = listOf(l1, l2)
}
