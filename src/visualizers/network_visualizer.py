"""
Instagram Network Analysis - Visualization Utilities

This module handles data visualization for Instagram network analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from datetime import datetime

from config.settings import VISUALIZATION
from config.logging_config import get_logger
from src.utils.storage import DataStorage

logger = get_logger('visualizers.network_visualizer')

class NetworkVisualizer:
    """
    Creates visualizations for Instagram network analysis.
    
    Features:
    - Network graphs of follower relationships
    - Distribution plots of metrics
    - Ranking visualizations
    - Export in multiple formats
    """
    
    def __init__(self, storage=None, base_dir=None):
        """
        Initialize the network visualizer.
        
        Args:
            storage (DataStorage, optional): Storage handler
            base_dir (str, optional): Base directory for storage
        """
        # Initialize storage
        if storage:
            self.storage = storage
        else:
            self.storage = DataStorage(base_dir=base_dir)
        
        # Visualization settings
        self.settings = VISUALIZATION
        
        # Set up matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        self.figsize = self.settings['DEFAULT_PLOT_SIZE']
    
    def create_network_graph(self, rankings, max_nodes=None, filename=None):
        """
        Create a network graph visualization.
        
        Args:
            rankings (dict): Rankings data
            max_nodes (int, optional): Maximum number of nodes to include
            filename (str, optional): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        if max_nodes is None:
            max_nodes = self.settings['MAX_NODES_IN_GRAPH']
        
        # Create a new figure
        plt.figure(figsize=self.figsize)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (top accounts by follower count)
        top_accounts = rankings['by_follower_count'][:max_nodes]
        
        for account in top_accounts:
            G.add_node(account['username'], 
                      size=account['follower_count'],
                      verified=account['is_verified'])
        
        # Add edges (connections between accounts)
        # This is a simplified version - in a real implementation, we would use actual following relationships
        # Here we're just connecting accounts that have similar follower counts
        for i, account1 in enumerate(top_accounts):
            for account2 in top_accounts[i+1:]:
                # Connect accounts with similar follower counts
                follower_diff = abs(account1['follower_count'] - account2['follower_count'])
                follower_max = max(account1['follower_count'], account2['follower_count'])
                
                if follower_diff / follower_max < 0.2:  # If difference is less than 20%
                    G.add_edge(account1['username'], account2['username'], 
                              weight=1 - (follower_diff / follower_max))
        
        # Calculate node sizes based on follower count
        sizes = [G.nodes[node]['size'] for node in G.nodes]
        max_size = max(sizes) if sizes else 1
        node_sizes = [100 + (1000 * (size / max_size)) for size in sizes]
        
        # Calculate node colors based on verified status
        node_colors = ['#1f77b4' if G.nodes[node]['verified'] else '#ff7f0e' for node in G.nodes]
        
        # Calculate edge widths based on weight
        edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges]
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        # Add legend
        plt.plot([], [], 'o', color='#1f77b4', label='Verified')
        plt.plot([], [], 'o', color='#ff7f0e', label='Not Verified')
        plt.legend()
        
        # Add title and adjust layout
        plt.title(f"Network of Top {len(G.nodes)} Instagram Accounts")
        plt.tight_layout()
        plt.axis('off')
        
        # Save the visualization
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"network_graph_{timestamp}"
        
        saved_paths = []
        for fmt in self.settings['EXPORT_FORMATS']:
            output_path = self.storage.base_dir / self.settings['LOCAL']['RESULTS_DIR'] / f"{filename}.{fmt}"
            plt.savefig(output_path, dpi=self.settings['DPI'], bbox_inches='tight')
            saved_paths.append(str(output_path))
        
        plt.close()
        
        logger.info(f"Network graph saved to {', '.join(saved_paths)}")
        return saved_paths[0] if saved_paths else None
    
    def create_follower_distribution(self, rankings, filename=None):
        """
        Create a follower distribution visualization.
        
        Args:
            rankings (dict): Rankings data
            filename (str, optional): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        # Create a new figure
        plt.figure(figsize=self.figsize)
        
        # Extract follower counts
        follower_counts = [account['follower_count'] for account in rankings['by_follower_count']]
        
        # Create histogram
        sns.histplot(follower_counts, bins=30, kde=True)
        
        # Add labels and title
        plt.xlabel('Number of Followers')
        plt.ylabel('Frequency')
        plt.title('Distribution of Follower Counts')
        
        # Add log scale for x-axis
        plt.xscale('log')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save the visualization
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"follower_distribution_{timestamp}"
        
        saved_paths = []
        for fmt in self.settings['EXPORT_FORMATS']:
            output_path = self.storage.base_dir / self.settings['LOCAL']['RESULTS_DIR'] / f"{filename}.{fmt}"
            plt.savefig(output_path, dpi=self.settings['DPI'], bbox_inches='tight')
            saved_paths.append(str(output_path))
        
        plt.close()
        
        logger.info(f"Follower distribution saved to {', '.join(saved_paths)}")
        return saved_paths[0] if saved_paths else None
    
    def create_top_accounts_bar_chart(self, rankings, metric='follower_count', top_n=20, filename=None):
        """
        Create a bar chart of top accounts by a specific metric.
        
        Args:
            rankings (dict): Rankings data
            metric (str): Metric to rank by ('follower_count', 'penetration_rate', or 'influence_score')
            top_n (int): Number of top accounts to include
            filename (str, optional): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        # Create a new figure
        plt.figure(figsize=self.figsize)
        
        # Determine which ranking to use
        if metric == 'follower_count':
            ranking_key = 'by_follower_count'
            title = f'Top {top_n} Accounts by Follower Count'
            xlabel = 'Number of Followers'
        elif metric == 'penetration_rate':
            ranking_key = 'by_penetration_rate'
            title = f'Top {top_n} Accounts by Penetration Rate'
            xlabel = 'Penetration Rate (%)'
        elif metric == 'influence_score':
            ranking_key = 'by_influence_score'
            title = f'Top {top_n} Accounts by Influence Score'
            xlabel = 'Influence Score'
        else:
            logger.error(f"Invalid metric: {metric}")
            return None
        
        # Extract data
        accounts = rankings[ranking_key][:top_n]
        usernames = [account['username'] for account in accounts]
        values = [account[metric] for account in accounts]
        
        # Create bar chart
        colors = ['#1f77b4' if account['is_verified'] else '#ff7f0e' for account in accounts]
        plt.barh(usernames[::-1], values[::-1], color=colors[::-1])
        
        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel('Username')
        plt.title(title)
        
        # Add legend
        plt.plot([], [], 'o', color='#1f77b4', label='Verified')
        plt.plot([], [], 'o', color='#ff7f0e', label='Not Verified')
        plt.legend(loc='lower right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the visualization
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"top_accounts_{metric}_{timestamp}"
        
        saved_paths = []
        for fmt in self.settings['EXPORT_FORMATS']:
            output_path = self.storage.base_dir / self.settings['LOCAL']['RESULTS_DIR'] / f"{filename}.{fmt}"
            plt.savefig(output_path, dpi=self.settings['DPI'], bbox_inches='tight')
            saved_paths.append(str(output_path))
        
        plt.close()
        
        logger.info(f"Top accounts bar chart saved to {', '.join(saved_paths)}")
        return saved_paths[0] if saved_paths else None
    
    def create_metric_comparison_scatter(self, rankings, filename=None):
        """
        Create a scatter plot comparing different metrics.
        
        Args:
            rankings (dict): Rankings data
            filename (str, optional): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        # Create a new figure
        plt.figure(figsize=self.figsize)
        
        # Extract data
        accounts = rankings['by_follower_count'][:100]  # Use top 100 accounts
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame([
            {
                'username': account['username'],
                'follower_count': account['follower_count'],
                'penetration_rate': account['penetration_rate'],
                'influence_score': account['influence_score'],
                'is_verified': account['is_verified']
            }
            for account in accounts
        ])
        
        # Create scatter plot
        sns.scatterplot(
            data=df,
            x='penetration_rate',
            y='influence_score',
            size='follower_count',
            hue='is_verified',
            sizes=(20, 500),
            alpha=0.7
        )
        
        # Add labels for top accounts
        for _, row in df.nlargest(10, 'influence_score').iterrows():
            plt.text(
                row['penetration_rate'] + 0.5,
                row['influence_score'] + 0.02,
                row['username'],
                fontsize=8
            )
        
        # Add labels and title
        plt.xlabel('Penetration Rate (%)')
        plt.ylabel('Influence Score')
        plt.title('Comparison of Penetration Rate vs. Influence Score')
        
        # Adjust legend
        plt.legend(title='Verified', loc='upper left')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save the visualization
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metric_comparison_{timestamp}"
        
        saved_paths = []
        for fmt in self.settings['EXPORT_FORMATS']:
            output_path = self.storage.base_dir / self.settings['LOCAL']['RESULTS_DIR'] / f"{filename}.{fmt}"
            plt.savefig(output_path, dpi=self.settings['DPI'], bbox_inches='tight')
            saved_paths.append(str(output_path))
        
        plt.close()
        
        logger.info(f"Metric comparison scatter plot saved to {', '.join(saved_paths)}")
        return saved_paths[0] if saved_paths else None
    
    def create_cluster_visualization(self, clusters, filename=None):
        """
        Create a visualization of account clusters.
        
        Args:
            clusters (dict): Cluster data
            filename (str, optional): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        # Create a new figure
        plt.figure(figsize=self.figsize)
        
        # Extract data
        cluster_sizes = []
        cluster_labels = []
        
        for cluster_id, cluster_data in clusters.items():
            cluster_sizes.append(cluster_data['size'])
            
            # Create label from top features
            top_features = cluster_data['top_features'][:3]  # Use top 3 features
            label = f"{cluster_id}: {', '.join(top_features)}"
            
            cluster_labels.append(label)
        
        # Create pie chart
        plt.pie(
            cluster_sizes,
            labels=cluster_labels,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            explode=[0.05] * len(cluster_sizes)
        )
        
        # Add title
        plt.title('Distribution of Account Clusters')
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')
        
        # Save the visualization
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cluster_visualization_{timestamp}"
        
        saved_paths = []
        for fmt in self.settings['EXPORT_FORMATS']:
            output_path = self.storage.base_dir / self.settings['LOCAL']['RESULTS_DIR'] / f"{filename}.{fmt}"
            plt.savefig(output_path, dpi=self.settings['DPI'], bbox_inches='tight')
            saved_paths.append(str(output_path))
        
        plt.close()
        
        logger.info(f"Cluster visualization saved to {', '.join(saved_paths)}")
        return saved_paths[0] if saved_paths else None
    
    def create_dashboard(self, rankings, clusters=None, filename=None):
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            rankings (dict): Rankings data
            clusters (dict, optional): Cluster data
            filename (str, optional): Output filename
            
        Returns:
            str: Path to saved dashboard
        """
        # Create a new figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Top accounts by follower count
        ax = axs[0, 0]
        accounts = rankings['by_follower_count'][:10]
        usernames = [account['username'] for account in accounts]
        values = [account['follower_count'] for account in accounts]
        colors = ['#1f77b4' if account['is_verified'] else '#ff7f0e' for account in accounts]
        
        ax.barh(usernames[::-1], values[::-1], color=colors[::-1])
        ax.set_xlabel('Number of Followers')
        ax.set_ylabel('Username')
        ax.set_title('Top 10 Accounts by Follower Count')
        
        # Add legend
        ax.plot([], [], 'o', color='#1f77b4', label='Verified')
        ax.plot([], [], 'o', color='#ff7f0e', label='Not Verified')
        ax.legend(loc='lower right')
        
        # 2. Follower distribution
        ax = axs[0, 1]
        follower_counts = [account['follower_count'] for account in rankings['by_follower_count']]
        
        sns.histplot(follower_counts, bins=30, kde=True, ax=ax)
        ax.set_xlabel('Number of Followers')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Follower Counts')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # 3. Top accounts by influence score
        ax = axs[1, 0]
        accounts = rankings['by_influence_score'][:10]
        usernames = [account['username'] for account in accounts]
        values = [account['influence_score'] for account in accounts]
        colors = ['#1f77b4' if account['is_verified'] else '#ff7f0e' for account in accounts]
        
        ax.barh(usernames[::-1], values[::-1], color=colors[::-1])
        ax.set_xlabel('Influence Score')
        ax.set_ylabel('Username')
        ax.set_title('Top 10 Accounts by Influence Score')
        
        # Add legend
        ax.plot([], [], 'o', color='#1f77b4', label='Verified')
        ax.plot([], [], 'o', color='#ff7f0e', label='Not Verified')
        ax.legend(loc='lower right')
        
        # 4. Cluster visualization or metric comparison
        ax = axs[1, 1]
        
        if clusters:
            # Extract data
            cluster_sizes = []
            cluster_labels = []
            
            for cluster_id, cluster_data in clusters.items():
                cluster_sizes.append(cluster_data['size'])
                
                # Create label from top features
                top_features = cluster_data['top_features'][:2]  # Use top 2 features
                label = f"{cluster_id}: {', '.join(top_features)}"
                
                cluster_labels.append(label)
            
            # Create pie chart
            ax.pie(
                cluster_sizes,
                labels=cluster_labels,
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                explode=[0.05] * len(cluster_sizes)
            )
            
            ax.set_title('Distribution of Account Clusters')
            ax.axis('equal')
        else:
            # Create metric comparison scatter plot
            accounts = rankings['by_follower_count'][:100]  # Use top 100 accounts
            
            # Create DataFrame for easier plotting
            df = pd.DataFrame([
                {
                    'username': account['username'],
                    'follower_count': account['follower_count'],
                    'penetration_rate': account['penetration_rate'],
                    'influence_score': account['influence_score'],
                    'is_verified': account['is_verified']
                }
                for account in accounts
            ])
            
            # Create scatter plot
            sns.scatterplot(
                data=df,
                x='penetration_rate',
                y='influence_score',
                size='follower_count',
                hue='is_verified',
                sizes=(20, 200),
                alpha=0.7,
                ax=ax
            )
            
            # Add labels for top accounts
            for _, row in df.nlargest(5, 'influence_score').iterrows():
                ax.text(
                    row['penetration_rate'] + 0.5,
                    row['influence_score'] + 0.02,
                    row['username'],
                    fontsize=8
                )
            
            ax.set_xlabel('Penetration Rate (%)')
            ax.set_ylabel('Influence Score')
            ax.set_title('Penetration Rate vs. Influence Score')
            ax.legend(title='Verified', loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(
            f"Instagram Network Analysis Dashboard\n"
            f"Target: {rankings['metadata']['target_username']} | "
            f"Followers Analyzed: {rankings['metadata']['total_followers_analyzed']} | "
            f"Unique Accounts: {rankings['metadata']['total_unique_accounts']}",
            fontsize=16
        )
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save the dashboard
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dashboard_{timestamp}"
        
        saved_paths = []
        for fmt in self.settings['EXPORT_FORMATS']:
            output_path = self.storage.base_dir / self.settings['LOCAL']['RESULTS_DIR'] / f"{filename}.{fmt}"
            plt.savefig(output_path, dpi=self.settings['DPI'], bbox_inches='tight')
            saved_paths.append(str(output_path))
        
        plt.close()
        
        logger.info(f"Dashboard saved to {', '.join(saved_paths)}")
        return saved_paths[0] if saved_paths else None
