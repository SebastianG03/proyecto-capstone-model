from typing import Optional
from app.entities.utils.singleton import Singleton
from app.shared.analysis_tools import AnalysisTools


class AnalysisContext(metaclass=Singleton):
    def __init__(self):
        self._tools: Optional[AnalysisTools] = None
    
    @property
    def tools(self) -> AnalysisTools:
        if self._tools is None:
            raise RuntimeError("AnalysisTools no ha sido inicializado aún.")
        return self._tools
    
    @tools.setter
    def tools(self, value: AnalysisTools) -> None:
        self._tools = value

analysis_context = AnalysisContext()