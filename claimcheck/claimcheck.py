"""ClaimCheck, a Reflex app. See state.py for events, views.py for screens and routes.py for media endpoints."""
import reflex as rx

from .routes import take_api
from .views import index

FONTS = "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&display=swap"
app = rx.App(
    theme=rx.theme(appearance="light", accent_color="gray", radius="none"),
    stylesheets=[FONTS, "/claimcheck.css"],
    api_transformer=take_api,
)
app.add_page(index, title="ClaimCheck", description="Film it. Say it. Sell what's true.")
