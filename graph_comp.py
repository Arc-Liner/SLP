import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.spatial import procrustes

def compute_procrustes_analysis(graph1, graph2):
    """
    Perform Procrustes analysis to align two 3D graphs.
    
    Args:
        graph1 (ndarray): 3D array (N x 3) for the first graph.
        graph2 (ndarray): 3D array (N x 3) for the second graph.
        
    Returns:
        float: Procrustes distance between the two graphs.
        ndarray: Transformed graph2 aligned to graph1.
    """
    # Ensure the inputs are 2D arrays with shape (N, 3)
    graph1 = graph1.reshape(-1, 3)
    graph2 = graph2.reshape(-1, 3)

    # Perform Procrustes analysis
    mtx1, mtx2, disparity = procrustes(graph1, graph2)
    
    return disparity, mtx2


def compute_mutual_information(graph1, graph2, bins=20):
    """
    Compute mutual information (MI) between two 3D graphs.
    
    Args:
        graph1 (ndarray): 3D array representing the first graph's values.
        graph2 (ndarray): 3D array representing the second graph's values.
        bins (int): Number of bins for the histogram.
        
    Returns:
        float: Estimated mutual information value.
    """
    # Flatten the graphs to 1D arrays for comparison
    graph1_flat = graph1.ravel()
    graph2_flat = graph2.ravel()
    
    # Joint histogram
    joint_hist, _, _ = np.histogram2d(graph1_flat, graph2_flat, bins=bins)
    
    # Normalize joint histogram to create a joint probability distribution
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Marginal probabilities
    p1 = np.sum(joint_prob, axis=1)  # Sum over rows
    p2 = np.sum(joint_prob, axis=0)  # Sum over columns
    
    # Compute mutual information
    mutual_info = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            if joint_prob[i, j] > 0:
                mutual_info += joint_prob[i, j] * np.log(joint_prob[i, j] / (p1[i] * p2[j]))
    
    return mutual_info


# Example usage
if __name__ == "__main__":
    # Generate synthetic 3D graph data
    np.random.seed(60)
    graph1 = np.random.rand(10, 10, 10)  # Random data for graph 1
    graph2 = graph1 + np.random.normal(0, 0.1, size=graph1.shape)  # Slightly noisy graph 2
    
    # Compute mutual information
    mi = compute_mutual_information(graph1, graph2, bins=30)
    print(f"Mutual Information: {mi:.4f}")

    graph3 = np.random.rand(10, 3)  # Random points in 3D
    graph4 = graph3 @ np.array([[0.9, -0.1, 0.0],
                                [0.1, 0.9, 0.0],
                                [0.0, 0.0, 1.0]])  # Rotated graph1

    graph4 += 0.5  # Add a bias to graph2
    
    # Perform Procrustes analysis
    distance, aligned_graph2 = compute_procrustes_analysis(graph3, graph4)
    
    print(f"Procrustes Distance: {distance:.4f}")
    print(f"Aligned Graph2:\n{aligned_graph2}")
