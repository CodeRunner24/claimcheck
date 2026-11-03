import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="claimcheck",
    disable_plugins=[SitemapPlugin],
    show_built_with_reflex=False,
)
