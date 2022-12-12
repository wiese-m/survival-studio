from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP

import example_setups as ex
from src.components.layout import create_layout

explainer = ex.setup_rsf_brca_explainer()
app = Dash(__name__, external_stylesheets=[BOOTSTRAP])
app.title = "Survival Studio"
app.layout = create_layout(app, explainer)
server = app.server


if __name__ == "__main__":
    app.run_server(debug=False)
