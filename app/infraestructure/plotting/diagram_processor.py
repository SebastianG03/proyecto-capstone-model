from app.infraestructure.plotting.drawers import HeatmapDrawer
from sqlalchemy.orm import Session


def generate_diagrams(db: Session) -> None:
    try:
        # DrawerFactory.run_drawer(HeatmapDrawer, db=db, positions=positions)
        HeatmapDrawer(db).draw_and_save()
    except Exception as e:
        print(f"Error drawing diagram: {e}")
