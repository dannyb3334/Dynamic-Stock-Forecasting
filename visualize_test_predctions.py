import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_and_process_data(file_path, lag):
    """
    Loads and processes the data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        lag (int): Lag value to filter the data.
    
    Returns:
        pd.Series: Time series.
        pd.Series: Ground truth values.
        pd.Series: Prediction values.
    """
    data = pd.read_csv(file_path)
    data = data[data.lag == lag]
    time = data['time']
    y = data['y']
    y_median = data['y_pred'] if 'y_pred' in data.columns else data['y_median']
    return time, y, y_median

def setup_plot(time, y, y_median):
    """
    Sets up the plot with labels, titles, and initial lines.
    
    Args:
        time (pd.Series): Time series.
        y (pd.Series): Ground truth values.
        y_median (pd.Series): Prediction values.
    
    Returns:
        tuple: Figure, axis, and line objects.
    """
    fig, ax = plt.subplots()
    line1, = ax.plot(time, y, label='Ground Truth (y)')
    line2, = ax.plot(time, y_median, label='Prediction (y_median)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Stock Trading Data')
    ax.legend()
    return fig, ax, line1, line2

def update_plot(frame, time, y, y_median, line1, line2, ax):
    """
    Updates the plot for the animation.
    
    Args:
        frame (int): Current frame number.
        time (pd.Series): Time series.
        y (pd.Series): Ground truth values.
        y_median (pd.Series): Prediction values.
        line1 (matplotlib.lines.Line2D): Line object for ground truth.
        line2 (matplotlib.lines.Line2D): Line object for predictions.
        ax (matplotlib.axes.Axes): Matplotlib axis.
    
    Returns:
        tuple: Updated line objects.
    """
    line1.set_data(time[:frame], y[:frame])
    line2.set_data(time[:frame], y_median[:frame])
    ax.relim()
    ax.autoscale_view()
    return line1, line2

def main():
    """
    Main function to execute the script.
    """
    # Parameters
    file_path = 'predictions.csv'
    lag = 60
    
    # Load and process data
    time, y, y_median = load_and_process_data(file_path, lag)
    
    # Setup plot
    fig, ax, line1, line2 = setup_plot(time, y, y_median)
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, 
        update_plot, 
        frames=len(time), 
        fargs=(time, y, y_median, line1, line2, ax), 
        interval=5, 
        blit=True
    )
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
