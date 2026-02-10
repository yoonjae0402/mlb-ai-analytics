export interface WalkthroughStep {
  page: string;
  selector: string;
  title: string;
  description: string;
  position?: "top" | "bottom" | "left" | "right";
}

const walkthroughSteps: WalkthroughStep[] = [
  {
    page: "/",
    selector: "[data-tour='hero']",
    title: "Welcome to MLB AI Analytics",
    description:
      "This platform uses machine learning to predict how MLB players will perform in their next game, using real Statcast data from the last several seasons.",
    position: "bottom",
  },
  {
    page: "/",
    selector: "[data-tour='features']",
    title: "Explore the Tools",
    description:
      "Each card takes you to a different tool — train models, visualize how AI thinks, predict player stats, and more.",
    position: "top",
  },
  {
    page: "/models",
    selector: "[data-tour='train-controls']",
    title: "Train AI Models",
    description:
      "Configure and train two different model types: LSTM (reads game sequences) and XGBoost (analyzes statistical patterns). Hit 'Train Models' to start.",
    position: "right",
  },
  {
    page: "/models",
    selector: "[data-tour='metrics']",
    title: "Compare Results",
    description:
      "After training, you'll see accuracy metrics here. Lower MSE and MAE = better predictions. Positive R² means the model learned something useful.",
    position: "bottom",
  },
  {
    page: "/predict",
    selector: "[data-tour='player-search']",
    title: "Search for a Player",
    description:
      "Type any MLB player's name to find them. The platform will load their recent stats and use AI to predict their next game performance.",
    position: "right",
  },
  {
    page: "/predict",
    selector: "[data-tour='predict-button']",
    title: "Make a Prediction",
    description:
      "Choose a model type and click 'Predict Next Game'. The AI analyzes the player's last 10 games to forecast hits, home runs, RBI, and walks.",
    position: "right",
  },
  {
    page: "/attention",
    selector: "[data-tour='attention-heatmap']",
    title: "See How the AI Thinks",
    description:
      "This heatmap shows which past games the AI paid most attention to when making its prediction. Brighter red = more important to the model.",
    position: "bottom",
  },
  {
    page: "/ensemble",
    selector: "[data-tour='ensemble-controls']",
    title: "Blend Models Together",
    description:
      "The Ensemble Lab lets you combine LSTM and XGBoost predictions. Drag the weight slider to find the best mix — often better than either model alone!",
    position: "bottom",
  },
];

export default walkthroughSteps;
