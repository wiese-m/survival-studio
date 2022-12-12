from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP

import example_setups as ex
from src.components.layout import create_layout


def main() -> None:
    explainer = ex.setup_rsf_brca_explainer()  # Choose one of the default explainer setups or make your own
    app = Dash(external_stylesheets=[BOOTSTRAP])
    server = app.server  # For webservice deployment
    app.title = "Survival Studio"
    app.layout = create_layout(app, explainer)
    app.run()


if __name__ == "__main__":
    main()
