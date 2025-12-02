package com.example.cartpolekt.simulation

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

data class CartPoleState(
    val x: Float = 0f,           // Cart position
    val xDot: Float = 0f,        // Cart velocity
    val theta: Float = 0f,       // Pole angle (radians)
    val thetaDot: Float = 0f     // Pole angular velocity
)

class CartPoleModel {
    private val gravity = 9.8f
    private val massCart = 1.0f
    private val massPole = 0.1f
    private val totalMass = massCart + massPole
    private val length = 0.5f // actually half the pole's length
    private val poleMassLength = massPole * length
    private val forceMag = 7.0f
    private val tau = 0.02f // seconds between state updates
    private val pi = PI.toFloat()
    private val twoPi = (2 * PI).toFloat()

    val xThreshold = 2.4f
    val thetaThresholdRadians = PI.toFloat() // allow full swing range
    
    private val xMax = 2.5f
    private val friction = 0.99f // Simple damping factor

    fun step(state: CartPoleState, action: Int): CartPoleState {
        // action: 0 for left, 1 for right, -1 for no force
        val force = when (action) {
            1 -> forceMag
            0 -> -forceMag
            2 -> 0f
            else -> 0f
        }
        
        val costheta = cos(state.theta)
        val sintheta = sin(state.theta)

        val temp = (force + poleMassLength * state.thetaDot * state.thetaDot * sintheta) / totalMass
        val thetaAcc = (gravity * sintheta - costheta * temp) / (length * (4.0f / 3.0f - massPole * costheta * costheta / totalMass))
        val xAcc = temp - poleMassLength * thetaAcc * costheta / totalMass

        var x = state.x + tau * state.xDot
        var xDot = (state.xDot + tau * xAcc) * friction // Apply friction
        val theta = wrapAngle(state.theta + tau * state.thetaDot)
        val thetaDot = (state.thetaDot + tau * thetaAcc) * friction // Apply friction

        // Wall collision (inelastic)
        if (x < -xMax) {
            x = -xMax
            xDot = 0f
        } else if (x > xMax) {
            x = xMax
            xDot = 0f
        }

        return state.copy(
            x = x,
            xDot = xDot,
            theta = theta,
            thetaDot = thetaDot
        )
    }
    
    private fun wrapAngle(angle: Float): Float {
        var a = angle
        while (a <= -pi) a += twoPi
        while (a > pi) a -= twoPi
        return a
    }
    
    fun reset(startUpright: Boolean = false): CartPoleState {
        val shouldStartUpright = startUpright
        val thetaNoise = Random.nextFloat() * 0.2f - 0.1f
        val baseTheta = if (shouldStartUpright) thetaNoise else wrapAngle(pi + thetaNoise)
        val xNoise = Random.nextFloat() * 0.1f - 0.05f
        val xDotNoise = Random.nextFloat() * 0.1f - 0.05f
        val thetaDotNoise = Random.nextFloat() * 0.1f - 0.05f
        return CartPoleState(
            x = xNoise,
            xDot = xDotNoise,
            theta = baseTheta,
            thetaDot = thetaDotNoise
        )
    }
}
