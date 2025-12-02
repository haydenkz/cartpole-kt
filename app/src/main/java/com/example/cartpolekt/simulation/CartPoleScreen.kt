package com.example.cartpolekt.simulation

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import com.example.cartpolekt.ui.theme.*
import kotlin.math.cos
import kotlin.math.sin

@Composable
fun CartPoleScreen(viewModel: CartPoleViewModel) {
    val state by viewModel.state.collectAsState()
    val mode by viewModel.mode.collectAsState()
    val episode by viewModel.episodeCount.collectAsState()
    val rewardHistory by viewModel.rewardHistory.collectAsState()

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            SimTopBar(episode = episode)
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            
            // 1. Mode Tabs
            ModeSelector(currentMode = mode, onModeSelected = { viewModel.setMode(it) })

            // 2. Reward Graph
            if (rewardHistory.isNotEmpty()) {
                RewardGraphCard(rewardHistory = rewardHistory)
            }

            // 3. Simulation Viewport
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(12.dp))
                    .background(SimulationGrid.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                    .clip(RoundedCornerShape(12.dp))
                    .padding(4.dp)
            ) {
                CartPoleCanvas(state = state)
                
                // Overlay Stats
                Text(
                    text = "x: %.2f  θ: %.2f".format(state.x, state.theta),
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(8.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
            }

            // 4. Controls
            if (mode == 0 || mode == 2) {
                ControlPanel(
                    onLeft = { viewModel.setLeftPressed(it) },
                    onRight = { viewModel.setRightPressed(it) },
                    onReset = { viewModel.reset() }
                )
            } else {
                // Placeholder spacing to keep layout stable
                Spacer(modifier = Modifier.height(80.dp))
            }
        }
    }
}

@Composable
fun SimTopBar(episode: Int) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp, horizontal = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = "CARTPOLE//SIM",
            style = MaterialTheme.typography.titleLarge,
            color = MaterialTheme.colorScheme.primary
        )
        
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Text(
                text = "EPISODE: $episode",
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun ModeSelector(currentMode: Int, onModeSelected: (Int) -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(50)),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        ModeTab("MANUAL", 0, currentMode, onModeSelected)
        ModeTab("TRAIN", 1, currentMode, onModeSelected)
        ModeTab("EVAL", 2, currentMode, onModeSelected)
    }
}

@Composable
fun ModeTab(text: String, index: Int, currentMode: Int, onSelect: (Int) -> Unit) {
    val selected = index == currentMode
    val color = if (selected) MaterialTheme.colorScheme.primary else Color.Transparent
    val contentColor = if (selected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface
    
    Button(
        onClick = { onSelect(index) },
        colors = ButtonDefaults.buttonColors(containerColor = color, contentColor = contentColor),
        shape = RoundedCornerShape(50),
        modifier = Modifier.padding(4.dp)
    ) {
        Text(text, style = MaterialTheme.typography.labelLarge)
    }
}

@Composable
fun RewardGraphCard(rewardHistory: List<Float>) {
    Card(
        modifier = Modifier.fillMaxWidth().height(120.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                "REWARD HISTORY", 
                style = MaterialTheme.typography.labelSmall, 
                color = MaterialTheme.colorScheme.tertiary
            )
            Spacer(modifier = Modifier.height(8.dp))
            RewardGraph(rewardHistory)
        }
    }
}

@Composable
fun RewardGraph(rewardHistory: List<Float>) {
    val primaryColor = MaterialTheme.colorScheme.tertiary
    
    Canvas(modifier = Modifier.fillMaxSize()) {
        if (rewardHistory.isEmpty()) return@Canvas
        
        val min = rewardHistory.minOrNull() ?: 0f
        val max = rewardHistory.maxOrNull() ?: 1f
        val range = if (max - min < 1f) 1f else max - min
        
        val path = Path()
        val widthPerPoint = size.width / (rewardHistory.size - 1).coerceAtLeast(1)
        
        rewardHistory.forEachIndexed { index, reward ->
            val x = index * widthPerPoint
            val normalizedReward = (reward - min) / range
            // 0 is bottom for graph visual
            val y = size.height - (normalizedReward * size.height)
            
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        
        drawPath(
            path = path,
            color = primaryColor,
            style = Stroke(width = 2.dp.toPx())
        )
    }
}

@Composable
fun ControlPanel(onLeft: (Boolean) -> Unit, onRight: (Boolean) -> Unit, onReset: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Left Arrow
        ControlKey("◄", onLeft)
        
        // Reset
        Button(
            onClick = onReset,
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.errorContainer,
                contentColor = MaterialTheme.colorScheme.onErrorContainer
            )
        ) {
            Text("RESET")
        }
        
        // Right Arrow
        ControlKey("►", onRight)
    }
}

@Composable
fun ControlKey(text: String, onHoldChanged: (Boolean) -> Unit) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()

    LaunchedEffect(isPressed) {
        onHoldChanged(isPressed)
    }

    OutlinedButton(
        onClick = {},
        interactionSource = interactionSource,
        modifier = Modifier.size(80.dp),
        shape = RoundedCornerShape(20.dp),
        border = if (isPressed) 
            ButtonDefaults.outlinedButtonBorder.copy(brush = Brush.linearGradient(listOf(CyberPrimary, CyberTertiary)))
            else ButtonDefaults.outlinedButtonBorder
    ) {
        Text(text, fontSize = 24.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun CartPoleCanvas(state: CartPoleState) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val canvasWidth = size.width
        val canvasHeight = size.height
        val centerX = canvasWidth / 2
        val centerY = canvasHeight / 2

        // Grid Background
        val gridSize = 40.dp.toPx()
        val gridColor = Color.White.copy(alpha = 0.05f)
        
        for (x in 0..canvasWidth.toInt() step gridSize.toInt()) {
            drawLine(gridColor, Offset(x.toFloat(), 0f), Offset(x.toFloat(), canvasHeight))
        }
        for (y in 0..canvasHeight.toInt() step gridSize.toInt()) {
            drawLine(gridColor, Offset(0f, y.toFloat()), Offset(canvasWidth, y.toFloat()))
        }

        // Physics Scale
        val scale = canvasWidth / 6.0f 

        // Rail & Walls
        val railY = centerY + 40.dp.toPx()
        val limitX = 2.5f
        val cartWidth = 80.dp.toPx()
        val cartHeight = 40.dp.toPx()
        val cartHalfWidth = cartWidth / 2
        
        // Visual Wall Positions: Physics Limit + Cart Half Width
        val wallLeft = centerX - (limitX * scale) - cartHalfWidth
        val wallRight = centerX + (limitX * scale) + cartHalfWidth
        val wallHeight = 50.dp.toPx()

        // Rail
        drawLine(
            color = CyberSurfaceBorder,
            start = Offset(wallLeft, railY),
            end = Offset(wallRight, railY),
            strokeWidth = 4.dp.toPx()
        )

        val cartX = centerX + (state.x * scale)
        val cartY = railY

        // Cart Drawing
        drawRoundRect(
            color = CartColor,
            topLeft = Offset(cartX - cartWidth / 2, cartY - cartHeight - 10.dp.toPx()), 
            size = Size(cartWidth, cartHeight),
            cornerRadius = CornerRadius(8.dp.toPx())
        )
        
        // Wheels
        val wheelRadius = 8.dp.toPx()
        drawCircle(WheelColor, wheelRadius, Offset(cartX - 25.dp.toPx(), cartY - 5.dp.toPx()))
        drawCircle(WheelColor, wheelRadius, Offset(cartX + 25.dp.toPx(), cartY - 5.dp.toPx()))

        // Pivot
        val pivotX = cartX
        val pivotY = cartY - cartHeight/2 - 10.dp.toPx()

        // Pole
        val poleLength = 150.dp.toPx()
        val poleWidth = 12.dp.toPx()
        
        rotate(degrees = Math.toDegrees(state.theta.toDouble()).toFloat(), pivot = Offset(pivotX, pivotY)) {
             drawRoundRect(
                 color = PoleColor,
                 topLeft = Offset(pivotX - poleWidth / 2, pivotY - poleLength),
                 size = Size(poleWidth, poleLength),
                 cornerRadius = CornerRadius(poleWidth / 2)
             )
             drawCircle(Color.Black, 4.dp.toPx(), Offset(pivotX, pivotY))
        }
    }
}
