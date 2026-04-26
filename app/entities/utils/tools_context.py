from typing import Optional
import app.entities.utils.singleton as singleton
import app.shared.analysis_tools as analysis_tools


class AnalysisContext(metaclass=singleton.Singleton):
    def __init__(self):
        self._tools: Optional[analysis_tools.AnalysisTools] = None

    @property
    def tools(self) -> analysis_tools.AnalysisTools:
        if self._tools is None:
            raise RuntimeError("AnalysisTools no ha sido inicializado aun.")
        return self._tools

    @tools.setter
    def tools(self, value: analysis_tools.AnalysisTools) -> None:
        self._tools = value


analysis_context = AnalysisContext()
