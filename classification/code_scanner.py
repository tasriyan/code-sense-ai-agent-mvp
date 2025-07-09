from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class CodeFile:
    """Represents a code file with metadata"""
    file_path: str
    project_name: str
    file_type: str  # 'cs' or 'appsettings'
    content: str
    relative_path: str

class CodeScanner:
    """Scans loyalty service projects for code files"""

    def __init__(self, project_root: str, projects: List[str]):
        self.project_root = Path(project_root)
        self.projects = projects

    def scan_files(self) -> List[CodeFile]:
        """Scan all projects for .cs and appsettings.json files"""
        code_files = []

        for project in self.projects:
            project_path = self.project_root / project.replace('.csproj', '')
            if project_path.exists():
                files = self._scan_project(project_path, project)
                code_files.extend(files)
                print(f"Scanned {len(files)} files from {project}")

        return code_files

    def _scan_project(self, project_path: Path, project_name: str) -> List[CodeFile]:
        """Scan a single project directory"""
        files = []

        # Scan for .cs files
        for cs_file in project_path.rglob("*.cs"):
            if self._should_include_file(cs_file):
                files.append(self._create_code_file(cs_file, project_name, 'cs'))

        # Scan for appsettings.json files
        for json_file in project_path.rglob("appsettings*.json"):
            files.append(self._create_code_file(json_file, project_name, 'appsettings'))

        return files

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included in scanning"""
        exclude_dirs = ['bin', 'obj', 'packages', '.git', '.vs']
        return not any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs)

    def _create_code_file(self, file_path: Path, project_name: str, file_type: str) -> CodeFile:
        """Create CodeFile object from file path"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Handle binary files or encoding issues
            content = ""

        return CodeFile(
            file_path=str(file_path),
            project_name=project_name,
            file_type=file_type,
            content=content,
            relative_path=str(file_path.relative_to(self.project_root))
        )