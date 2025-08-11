# 📁 app/main.py (modularer Aufbau mit gleichen Endpunkten)
from fastapi import FastAPI
from api.routes import process_routes, upload_routes, team_routes, result_routes

app = FastAPI()

# ⬇️ Routen modular einbinden
app.include_router(upload_routes.router)
app.include_router(process_routes.router)
app.include_router(team_routes.router)
app.include_router(result_routes.router)