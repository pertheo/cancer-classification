import matplotlib.pyplot as plt

def plot_roc_curve_rf(fper, tper):  
    plt.plot(fper, tper, color='green', label='Random Forest')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

def plot_roc_curve_lr(fper, tper):  
    plt.plot(fper, tper, color='red', label='Logistic Regression')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def plot_roc_curve_gb(fper, tper):  
    plt.plot(fper, tper, color='blue', label='Gradient Boosting')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
