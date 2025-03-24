import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ( mean_squared_error, r2_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, confusion_matrix)

#create class Report
class Report:
    #initialize with model,x test ,y test
    def __init__(self, model, X_test, y_test, task="auto"):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.task = self.task(task)
        self.y_pred = self._get_predictions()
    # function that finds out task is regression or classification(we have three tasks)
    def task(self, task):
        
        if task != "auto":
            return task  # Use user-defined task

        if len(np.unique(self.y_test)) > 2:
            return "classification"
        elif np.issubdtype(self.y_test.dtype, np.integer):
            return "classification"
        else:
            return "regression"
    #Generate predictions and correctly format outputs based on task.
    def _get_predictions(self):
        
        if self.model is None or self.X_test is None:
            return None
        
        y_pred = self.model.forward(self.X_test)

        # Handle classification tasks (Logistic Regression & MLP)
        if self.task == "classification":
            if y_pred.shape[1] > 1:  # Multi-class (MLP with softmax)
                return np.argmax(y_pred, axis=1)
            else:  # Binary Classification (Logistic Regression)
                return (y_pred > 0.5).astype(int)

        # Regression case (keep output as continuous values)
        return y_pred
     #calculate and print regression metrics for Linear Regression.
    def regression_metrics(self):
       
        if self.y_pred is None:
            print("Error: No predictions available.")
            return
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R2): {r2:.4f}")

    def plot_parity(self):
        """Plot parity plot for regression."""
        if self.y_pred is None:
            print("Error: No predictions available.")
            return
        plt.scatter(self.y_test, self.y_pred, alpha=0.5)
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Parity Plot')
        plt.show()
    #Plot confusion matrix for classification.
    def plot_confusion_matrix(self):
        
        if self.task != "classification":
            print("Error: Confusion matrix only applies to classification tasks.")
            return
        if self.y_pred is None:
            print("Error: No predictions available.")
            return
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
     #  calculate and print classification metrics.
    def classification_metrics(self):
        
        if self.task != "classification":
            print("Error: Classification metrics only apply to classification tasks.")
            return
        if self.y_pred is None:
            print("Error: No predictions available.")
            return
        
        print(f"Accuracy: {accuracy_score(self.y_test, self.y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, self.y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(self.y_test, self.y_pred, average='weighted'):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, self.y_pred, average='weighted'):.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))
    

    #function that generate evaluation report for all task
    def generate_report(self):
        
        if self.y_pred is None:
            print("Error: No predictions available.")
            return
        
        if self.task == 'regression':
            self.regression_metrics()
            self.plot_parity()
        else:
            self.classification_metrics()
            self.plot_confusion_matrix()


    #function to get loss_curve: training loss and validation loss
    def plot_loss_curve(self, train_losses, val_losses):
        """Plot training vs. validation loss curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs. Validation Loss Curve')
        plt.legend()
        plt.show()

