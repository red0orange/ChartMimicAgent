try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Set the figure size
    plt.figure(figsize=(10.0, 7.0))
    
    # Define the grid
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    
    # Define the functions
    Z1 = X**2 - Y**2  # Saddle-like surface for f(x)
    Z2 = np.exp(-(X**2 + Y**2))  # Peak-like surface for the solution
    
    # Create subplots
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122, projection='3d')
    
    # Plot the first surface (f(x))
    surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', edgecolor='none')
    ax1.set_title('f(x)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.view_init(elev=30, azim=-60)  # Set viewing angle
    
    # Plot the second surface (solution)
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='viridis', edgecolor='none')
    ax2.set_title('solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.view_init(elev=30, azim=-60)  # Set viewing angle
    
    # Add a color bar which maps values to colors
    fig = plt.gcf()
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
except Exception as e:
    exit(100)
plt.savefig("temp_6b40480c.pdf")