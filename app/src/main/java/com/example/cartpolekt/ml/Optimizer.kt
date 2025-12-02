package com.example.cartpolekt.ml

import kotlin.math.pow
import kotlin.math.sqrt

// Optimizer for our custom layers
class AdamOptimizer(private val lr: Float = 0.001f, private val beta1: Float = 0.9f, private val beta2: Float = 0.999f, private val epsilon: Float = 1e-8f) {
    
    fun step(layers: List<DenseLayer>) {
        for (layer in layers) {
            layer.t++
            
            // Update Weights
            for (i in layer.weights.indices) {
                for (j in layer.weights[i].indices) {
                    val g = layer.weightGradients[i][j]
                    layer.mW[i][j] = beta1 * layer.mW[i][j] + (1 - beta1) * g
                    layer.vW[i][j] = beta2 * layer.vW[i][j] + (1 - beta2) * g * g
                    
                    val mHat = layer.mW[i][j] / (1 - beta1.pow(layer.t))
                    val vHat = layer.vW[i][j] / (1 - beta2.pow(layer.t))
                    
                    layer.weights[i][j] -= lr * mHat / (sqrt(vHat) + epsilon)
                    layer.weightGradients[i][j] = 0f // Reset gradient
                }
            }

            // Update Biases
            for (i in layer.biases.indices) {
                val g = layer.biasGradients[i]
                layer.mB[i] = beta1 * layer.mB[i] + (1 - beta1) * g
                layer.vB[i] = beta2 * layer.vB[i] + (1 - beta2) * g * g
                
                val mHat = layer.mB[i] / (1 - beta1.pow(layer.t))
                val vHat = layer.vB[i] / (1 - beta2.pow(layer.t))
                
                layer.biases[i] -= lr * mHat / (sqrt(vHat) + epsilon)
                layer.biasGradients[i] = 0f // Reset gradient
            }
        }
    }
}
