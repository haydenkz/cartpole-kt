package com.example.cartpolekt.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = CyberPrimary,
    secondary = CyberSecondary,
    tertiary = CyberTertiary,
    background = CyberDarkBackground,
    surface = CyberSurface,
    onPrimary = Color.Black,
    onSecondary = Color.Black,
    onTertiary = Color.Black,
    onBackground = CyberOnBackground,
    onSurface = CyberOnBackground,
)

@Composable
fun CartpoleKTTheme(
    content: @Composable () -> Unit
) {
    // Enforce Dark Theme for the Sci-Fi look
    MaterialTheme(
        colorScheme = DarkColorScheme,
        typography = Typography,
        content = content
    )
}
