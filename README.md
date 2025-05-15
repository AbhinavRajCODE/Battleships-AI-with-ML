🔱 Battleship AI – Modern Pygame Edition
Battleship AI – Modern Pygame Edition is a beautifully designed, turn-based strategy game built using Python and Pygame. It modernizes the classic Battleship gameplay with a sleek user interface, smooth visuals, and an intelligent AI opponent that learns from your playstyle using a Random Forest classifier. Whether you're looking to test your strategic thinking or explore how AI adapts in games, this project brings it all together in an accessible and polished package.

🎮 Game Features
🧠 Adaptive AI
The enemy isn't just guessing — it learns. Powered by machine learning, the AI opponent uses a Random Forest model to predict where your ships might be based on previous patterns. This adds a layer of challenge and realism to the gameplay.

🎨 Modern UI
The game features a redesigned, minimalist user interface:

Rounded board edges

Custom color palette (navy blue, light blue, cream, orange, and more)

Animated transitions for round changes

Stylized ship placement hints and status messages

Enhanced readability using dynamic font scaling

📊 Visual Enhancements

Clean grid layout with smooth grid lines

Hit, miss, and ship indicators with vivid, contrasting colors

Clear indicators for turns, scores, and game phase

Subtle animations and transparency effects for cell highlights

🎯 Gameplay Mechanics

10x10 grid traditional Battleship rules

Place your fleet with click + rotate ('R' key) functionality

Take turns firing at the enemy grid

First to win the majority out of multiple rounds wins the match

📈 Intelligent Feedback Loop

AI updates its training data each round

Learns your patterns over time to increase difficulty

Balanced win/loss system to keep gameplay fair and exciting

🛠️ Tech Stack

Python 3.x

Pygame for graphics and game loop

Scikit-learn for machine learning (Random Forest)

Numpy for data processing and efficient grid management

⚙️ How It Works
Placement Phase
Players place their ships on the grid. Orientation can be changed using the 'R' key. The UI highlights your ship size and placement instructions dynamically.

Battle Phase
Players and AI take turns guessing the location of each other’s ships. Hits and misses are shown instantly. The game automatically checks for sunk ships.

Round System
After each round, scores are updated. A brief transition shows the round number and resets the boards. The AI uses the previous round’s data to retrain itself, improving over time.

Victory
After the set number of rounds (e.g., best of 5), the game declares a winner and displays the final score.

🧪 AI Strategy
The AI builds a feature set based on past ship locations, hits, and misses.

It uses a Random Forest Classifier to identify the most probable coordinates for your ships.

With each round, its prediction accuracy improves, making the game more challenging.

🖼️ UI & Style Details
Colors

Navy Blue background for depth

Cream and Orange for highlights and messages

Light Blue for misses, Dark Red for hits

Rounded rectangles and alpha-blended surfaces for modern look

Fonts

Dynamically sized for titles, messages, and instructions

Clear visual hierarchy: large titles, medium subheadings, small instructions

Animations

Smooth screen refresh

Fading round transition messages

Alpha blending on ship/miss/hit overlays

🚀 Ideal For
Students learning Python, Pygame, or machine learning

Developers experimenting with AI in games

Fans of Battleship who want a smarter, prettier version of the game

Game developers looking to build projects with adaptive difficulty
