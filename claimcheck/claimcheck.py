"""ClaimCheck, a Reflex app. See state.py for events, views.py for screens and routes.py for media endpoints."""
import reflex as rx

from .routes import take_api
from .views import index

FONTS = "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&display=swap"
app = rx.App(
    theme=rx.theme(appearance="light", accent_color="gray", radius="none"),
    stylesheets=[FONTS, "/claimcheck.css"],
    api_transformer=take_api,
    head_components=[rx.el.link(rel="icon", href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' fill='%23ffdd00'/><path d='M3 8.5l3 3 7-7' stroke='%23111' stroke-width='2.4' fill='none'/></svg>")],
)
app.add_page(index, title="ClaimCheck", description="Film it. Say it. Sell what's true.")
