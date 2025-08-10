from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class REPLInput(BaseModel):
    code: str = Field(description="Python code to execute")


class REPLTool(BaseTool):
    name: str = "python_repl"
    description: str = "Execute Python code for calculations, data analysis, or testing algorithms. Returns the output or any errors."
    args_schema: Type[BaseModel] = REPLInput
    
    def _run(self, code: str) -> str:
        """Execute Python code and return output."""
        repl = PythonREPL()
        try:
            # If code doesn't have print, wrap last expression to capture output
            lines = code.strip().split('\n')
            if lines and not any('print' in line for line in lines):
                # Check if last line is an expression (not assignment or import)
                last_line = lines[-1].strip()
                if (last_line and 
                    not last_line.startswith(('import ', 'from ', 'def ', 'class ')) and
                    '=' not in last_line):
                    # Wrap last expression in print
                    lines[-1] = f"print({last_line})"
                    code = '\n'.join(lines)
            
            result = repl.run(code)
            return result if result else "Code executed (no output captured)"
        except Exception as e:
            return f"Error: {str(e)}"