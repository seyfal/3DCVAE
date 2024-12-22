# anomaly_detection/utils/latent_space_viz.py.py
# Author: Seyfal Sultanov 

import io
import base64
import numpy as np
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import torch
from tqdm import tqdm
import random
import logging
import time

logger = logging.getLogger(__name__)

def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="png")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/png;base64, " + encoded_image
    return im_url

def latent_space_visualization(model, data_loader, config, epoch, max_samples=5000, methods=['umap', 'tsne', 'pca']):
    """
    Generate interactive 3D visualizations of the latent space.
    
    Args:
        model (torch.nn.Module): The trained model
        data_loader (torch.utils.data.DataLoader): DataLoader containing the data
        config (dict): Configuration dictionary
        epoch (int): Current epoch number
        max_samples (int): Maximum number of samples to use for visualization
        methods (list): List of visualization methods to use ('umap', 'tsne', 'pca')
    
    Returns:
        Dash: A Dash app object that can be run to view the visualization
    """
    logger.info("Starting latent space visualization")
    device = torch.device(config['device'])
    model.eval()
    
    # Randomly select samples
    total_samples = len(data_loader.dataset)
    random_indices = random.sample(range(total_samples), min(max_samples, total_samples))
    
    # Collect embeddings and original data
    embeddings = []
    original_data = []
    
    logger.info("Collecting embeddings and original data")
    with torch.no_grad():
        for idx in tqdm(random_indices, desc="Collecting random samples"):
            x = data_loader.dataset[idx]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            x = x.unsqueeze(0).to(device)
            
            mean, _ = model.encode(x)
            embeddings.append(mean.cpu().numpy())
            original_data.append(x.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    original_data = np.vstack(original_data)
    
    # Perform dimensionality reduction
    reduced_embeddings = {}
    for method in methods:
        logger.info(f"Performing {method.upper()} dimensionality reduction")
        start_time = time.time()
        if method == 'umap':
            reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
            reduced_embeddings[method] = reducer.fit_transform(embeddings)
        elif method == 'tsne':
            tsne = TSNE(n_components=3, random_state=42, verbose=1)
            reduced_embeddings[method] = tsne.fit_transform(embeddings)
        elif method == 'pca':
            pca = PCA(n_components=3)
            reduced_embeddings[method] = pca.fit_transform(embeddings)
        end_time = time.time()
        logger.info(f"{method.upper()} completed in {end_time - start_time:.2f} seconds")
    
    # Create Dash app
    app = Dash(__name__)
    
    # Create figures for each method
    figures = {}
    for method, embedding in reduced_embeddings.items():
        fig = go.Figure(data=[go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=np.sum(original_data, axis=(1, 2, 3, 4)),
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=f"3D {method.upper()} of Latent Space (Epoch {epoch})",
            scene=dict(
                xaxis_title=f"{method.upper()} 1",
                yaxis_title=f"{method.upper()} 2",
                zaxis_title=f"{method.upper()} 3",
            ),
        )
        
        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
        )
        
        figures[method] = fig
    
    # Create layout
    app.layout = html.Div([
        dcc.Dropdown(
            id='method-selector',
            options=[{'label': method.upper(), 'value': method} for method in methods],
            value=methods[0],
            style={'width': '200px'}
        ),
        dcc.Graph(id="graph-3d-plot", figure=figures[methods[0]], clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip", direction='bottom'),
    ])
    
    @app.callback(
        Output("graph-3d-plot", "figure"),
        Input("method-selector", "value")
    )
    def update_graph(selected_method):
        return figures[selected_method]
    
    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-3d-plot", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update
    
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]
    
        # Create EELS image
        eels_data = original_data[num].squeeze().sum(axis=-1)
        eels_data = (eels_data - eels_data.min()) / (eels_data.max() - eels_data.min())
        eels_image = (eels_data * 255).astype(np.uint8)
        im_url = np_image_to_base64(eels_image)
    
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "100px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P(f"Sample {num}", style={'font-weight': 'bold', 'text-align': 'center'})
            ])
        ]
    
        return True, bbox, children
    
    logger.info("Latent space visualization prepared")
    return app

# Usage example:
# app = latent_space_visualization(model, data_loader, config, epoch, max_samples=5000, methods=['umap', 'tsne', 'pca'])
# app.run_server(debug=True)