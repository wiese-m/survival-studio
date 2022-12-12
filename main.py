from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP

import example_setups as ex
from src.components.layout import create_layout


def main() -> None:
    explainer = ex.setup_rsf_brca_explainer()  # Choose one of the default explainer setups or make your own
    app = Dash(__name__, external_stylesheets=[BOOTSTRAP])
    app.title = "Survival Studio"
    app.layout = create_layout(app, explainer)

    # Webservice deployment
    application = app
    server = application.server

    app.run()


if __name__ == "__main__":
    main()
