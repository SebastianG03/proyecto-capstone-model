from typing import Dict

from app.modules.plotting.drawers import HeatmapDrawer
from sqlalchemy.orm import Session

def generate_diagrams(db: Session, positions: Dict) -> None:
    try:
        # DrawerFactory.run_drawer(HeatmapDrawer, db=db, positions=positions)
        HeatmapDrawer(db, positions).draw_and_save()
    except Exception as e:
        print(f"Error drawing diagram: {e}")
