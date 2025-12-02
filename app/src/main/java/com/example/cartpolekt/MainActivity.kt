package com.example.cartpolekt

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import com.example.cartpolekt.simulation.CartPoleScreen
import com.example.cartpolekt.simulation.CartPoleViewModel
import com.example.cartpolekt.ui.theme.CartpoleKTTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val viewModel = CartPoleViewModel() // Simple instantiation for this scope
        setContent {
            CartpoleKTTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Box(modifier = Modifier.padding(innerPadding)) {
                        CartPoleScreen(viewModel = viewModel)
                    }
                }
            }
        }
    }
}
