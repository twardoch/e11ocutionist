#!/usr/bin/env python3
# this_file: src/e11ocutionist/e11ocutionist.py
"""
E11ocutionist: A multi-stage document processing pipeline for transforming literary content
into enhanced speech synthesis markup.

This module serves as the main orchestrator for the processing pipeline.

Created by Adam Twardoch
"""

import json
import shutil
import datetime
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from loguru import logger


# Define processing steps as an enum
class ProcessingStep(Enum):
    """Processing steps in the e11ocutionist pipeline."""

    CHUNKING = auto()
    ENTITIZING = auto()
    ORATING = auto()
    TONING_DOWN = auto()
    ELEVENLABS_CONVERSION = auto()


@dataclass
class PipelineConfig:
    """Configuration for the e11ocutionist pipeline."""

    # Input and output
    input_file: Path
    output_dir: Path | None = None

    # Step configuration
    start_step: ProcessingStep = ProcessingStep.CHUNKING
    force_restart: bool = False
    backup: bool = False

    # Model configuration
    chunker_model: str = "gpt-4"
    chunker_temperature: float = 0.2
    entitizer_model: str = "gpt-4"
    entitizer_temperature: float = 0.1
    orator_model: str = "gpt-4"
    orator_temperature: float = 0.7
    tonedown_model: str = "gpt-4"
    tonedown_temperature: float = 0.1

    # Processing options
    orator_all_steps: bool = True
    orator_sentences: bool = False
    orator_words: bool = False
    orator_punctuation: bool = False
    orator_emotions: bool = False
    min_em_distance: int | None = None
    dialog_mode: bool = True
    plaintext_mode: bool = False

    # Other options
    verbose: bool = False
    debug: bool = False

    # Other fields will be added as needed


class E11ocutionistPipeline:
    """Main pipeline orchestrator for e11ocutionist."""

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with the given configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.progress: dict[str, Any] = {}
        self.progress_file: Path | None = None

        # Set up logger
        if config.debug:
            logger.remove()
            logger.add(
                lambda msg: print(msg),
                level="DEBUG",
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            )
        elif config.verbose:
            logger.remove()
            logger.add(
                lambda msg: print(msg),
                level="INFO",
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            )

        # Set up output directory
        self._setup_output_directory()

    def _setup_output_directory(self) -> None:
        """Set up the output directory and progress file."""
        input_stem = self.config.input_file.stem

        if self.config.output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.output_dir = Path(f"{input_stem}_{timestamp}")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.config.output_dir / "progress.json"

        logger.info(f"Output directory: {self.config.output_dir}")

        # Initialize or load progress
        self._load_progress()

    def _load_progress(self) -> None:
        """Load progress from the progress file if it exists."""
        if (
            self.progress_file
            and self.progress_file.exists()
            and not self.config.force_restart
        ):
            try:
                with open(self.progress_file) as f:
                    self.progress = json.load(f)
                logger.info(f"Loaded progress from {self.progress_file}")
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
                self.progress = {}
        else:
            self.progress = {}

    def _save_progress(self) -> None:
        """Save progress to the progress file."""
        if self.progress_file:
            try:
                with open(self.progress_file, "w") as f:
                    json.dump(self.progress, f, indent=2)
                logger.debug(f"Saved progress to {self.progress_file}")
            except Exception as e:
                logger.error(f"Error saving progress: {e}")

    def _create_backup(self, file_path: Path) -> None:
        """Create a backup of the specified file.

        Args:
            file_path: Path to the file to back up
        """
        if self.config.backup and file_path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_name(
                f"{file_path.stem}_{timestamp}{file_path.suffix}"
            )
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")

    def run(self) -> dict[str, Any]:
        """Run the e11ocutionist pipeline.

        Returns:
            Dictionary with processing results and summary
        """
        logger.info("Starting e11ocutionist pipeline")

        # Track what steps were executed in this run
        executed_steps = []

        # Process each step sequentially
        if self._should_run_step(ProcessingStep.CHUNKING):
            self._run_chunking()
            executed_steps.append("chunking")

        if self._should_run_step(ProcessingStep.ENTITIZING):
            self._run_entitizing()
            executed_steps.append("entitizing")

        if self._should_run_step(ProcessingStep.ORATING):
            self._run_orating()
            executed_steps.append("orating")

        if self._should_run_step(ProcessingStep.TONING_DOWN):
            self._run_toning_down()
            executed_steps.append("toning_down")

        if self._should_run_step(ProcessingStep.ELEVENLABS_CONVERSION):
            self._run_elevenlabs_conversion()
            executed_steps.append("elevenlabs_conversion")

        # Create summary
        summary = {
            "input_file": str(self.config.input_file),
            "output_dir": str(self.config.output_dir),
            "executed_steps": executed_steps,
            "progress": self.progress,
        }

        # Save summary to file
        summary_file = self.config.output_dir / "_summary.txt"
        with open(summary_file, "w") as f:
            f.write("E11ocutionist Pipeline Summary\n")
            f.write("============================\n\n")
            f.write(f"Input file: {self.config.input_file}\n")
            f.write(f"Output directory: {self.config.output_dir}\n\n")
            f.write(f"Executed steps: {', '.join(executed_steps)}\n\n")
            f.write("Step details:\n")
            for step, details in self.progress.items():
                if isinstance(details, dict):
                    f.write(f"  {step}:\n")
                    for key, value in details.items():
                        f.write(f"    {key}: {value}\n")
                else:
                    f.write(f"  {step}: {details}\n")

        logger.info(f"Pipeline completed. Summary saved to {summary_file}")
        return summary

    def _should_run_step(self, step: ProcessingStep) -> bool:
        """Determine if a step should be run based on start_step and progress.

        Args:
            step: The step to check

        Returns:
            True if the step should be run, False otherwise
        """
        # If force_restart, always run if step is at or after start_step
        if self.config.force_restart:
            return step.value >= self.config.start_step.value

        # Otherwise, check if step is completed in progress
        step_name = step.name.lower()
        if step_name in self.progress and self.progress[step_name].get(
            "completed", False
        ):
            return False

        # If step is at or after start_step, run it
        return step.value >= self.config.start_step.value

    def _run_chunking(self) -> None:
        """Run the chunking step."""
        from .chunker import process_document

        logger.info("Running chunking step")

        step_name = ProcessingStep.CHUNKING.name.lower()
        input_file = self.config.input_file
        output_file = self.config.output_dir / f"{input_file.stem}_step1_chunked.xml"

        # Create backup if needed
        self._create_backup(output_file)

        # Run the chunking process
        try:
            result = process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                model=self.config.chunker_model,
                temperature=self.config.chunker_temperature,
                verbose=self.config.verbose,
                backup=self.config.backup,
            )

            # Update progress
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": True,
                "result": result,
            }
            self._save_progress()

            logger.info(f"Chunking completed: {output_file}")

        except Exception as e:
            logger.error(f"Error in chunking step: {e}")
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": False,
                "error": str(e),
            }
            self._save_progress()
            raise

    def _run_entitizing(self) -> None:
        """Run the entitizing step."""
        from .entitizer import process_document

        logger.info("Running entitizing step")

        step_name = ProcessingStep.ENTITIZING.name.lower()
        prev_step = ProcessingStep.CHUNKING.name.lower()

        if prev_step in self.progress and self.progress[prev_step].get(
            "completed", False
        ):
            input_file = Path(self.progress[prev_step]["output_file"])
        else:
            msg = "Cannot run entitizing: previous step (chunking) not completed"
            raise ValueError(msg)

        output_file = (
            self.config.output_dir
            / f"{self.config.input_file.stem}_step2_entitized.xml"
        )

        # Create backup if needed
        self._create_backup(output_file)

        # Run the entitizing process
        try:
            result = process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                model=self.config.entitizer_model,
                temperature=self.config.entitizer_temperature,
                verbose=self.config.verbose,
                backup=self.config.backup,
            )

            # Update progress
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": True,
                "result": result,
            }
            self._save_progress()

            logger.info(f"Entitizing completed: {output_file}")

        except Exception as e:
            logger.error(f"Error in entitizing step: {e}")
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": False,
                "error": str(e),
            }
            self._save_progress()
            raise

    def _run_orating(self) -> None:
        """Run the orating step."""
        from .orator import process_document

        logger.info("Running orating step")

        step_name = ProcessingStep.ORATING.name.lower()
        prev_step = ProcessingStep.ENTITIZING.name.lower()

        if prev_step in self.progress and self.progress[prev_step].get(
            "completed", False
        ):
            input_file = Path(self.progress[prev_step]["output_file"])
        else:
            msg = "Cannot run orating: previous step (entitizing) not completed"
            raise ValueError(msg)

        output_file = (
            self.config.output_dir / f"{self.config.input_file.stem}_step3_orated.xml"
        )

        # Create backup if needed
        self._create_backup(output_file)

        # Determine which orating steps to run
        if self.config.orator_all_steps:
            steps = ["--all_steps"]
        else:
            steps = []
            if self.config.orator_sentences:
                steps.append("--sentences")
            if self.config.orator_words:
                steps.append("--words")
            if self.config.orator_punctuation:
                steps.append("--punctuation")
            if self.config.orator_emotions:
                steps.append("--emotions")

        # Run the orating process
        try:
            result = process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                model=self.config.orator_model,
                temperature=self.config.orator_temperature,
                verbose=self.config.verbose,
                backup=self.config.backup,
                steps=steps,
            )

            # Update progress
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": True,
                "result": result,
                "steps": steps,
            }
            self._save_progress()

            logger.info(f"Orating completed: {output_file}")

        except Exception as e:
            logger.error(f"Error in orating step: {e}")
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": False,
                "error": str(e),
                "steps": steps,
            }
            self._save_progress()
            raise

    def _run_toning_down(self) -> None:
        """Run the toning down step."""
        from .tonedown import process_document

        logger.info("Running toning down step")

        step_name = ProcessingStep.TONING_DOWN.name.lower()
        prev_step = ProcessingStep.ORATING.name.lower()

        if prev_step in self.progress and self.progress[prev_step].get(
            "completed", False
        ):
            input_file = Path(self.progress[prev_step]["output_file"])
        else:
            msg = "Cannot run toning down: previous step (orating) not completed"
            raise ValueError(msg)

        output_file = (
            self.config.output_dir
            / f"{self.config.input_file.stem}_step4_toneddown.xml"
        )

        # Create backup if needed
        self._create_backup(output_file)

        # Run the toning down process
        try:
            em_args = (
                {"min_distance": self.config.min_em_distance}
                if self.config.min_em_distance
                else {}
            )

            result = process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                model=self.config.tonedown_model,
                temperature=self.config.tonedown_temperature,
                verbose=self.config.verbose,
                **em_args,
            )

            # Update progress
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": True,
                "result": result,
                "em_min_distance": self.config.min_em_distance,
            }
            self._save_progress()

            logger.info(f"Toning down completed: {output_file}")

        except Exception as e:
            logger.error(f"Error in toning down step: {e}")
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": False,
                "error": str(e),
                "em_min_distance": self.config.min_em_distance,
            }
            self._save_progress()
            raise

    def _run_elevenlabs_conversion(self) -> None:
        """Run the ElevenLabs conversion step."""
        from .elevenlabs_converter import process_document

        logger.info("Running ElevenLabs conversion step")

        step_name = ProcessingStep.ELEVENLABS_CONVERSION.name.lower()
        prev_step = ProcessingStep.TONING_DOWN.name.lower()

        if prev_step in self.progress and self.progress[prev_step].get(
            "completed", False
        ):
            input_file = Path(self.progress[prev_step]["output_file"])
        else:
            msg = "Cannot run ElevenLabs conversion: previous step (toning down) not completed"
            raise ValueError(msg)

        output_file = (
            self.config.output_dir / f"{self.config.input_file.stem}_step5_11labs.txt"
        )

        # Create backup if needed
        self._create_backup(output_file)

        # Run the ElevenLabs conversion process
        try:
            result = process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                dialog=self.config.dialog_mode,
                plaintext=self.config.plaintext_mode,
                verbose=self.config.verbose,
            )

            # Update progress
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": True,
                "result": result,
                "dialog": self.config.dialog_mode,
                "plaintext": self.config.plaintext_mode,
            }
            self._save_progress()

            logger.info(f"ElevenLabs conversion completed: {output_file}")

        except Exception as e:
            logger.error(f"Error in ElevenLabs conversion step: {e}")
            self.progress[step_name] = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp": datetime.datetime.now().isoformat(),
                "completed": False,
                "error": str(e),
                "dialog": self.config.dialog_mode,
                "plaintext": self.config.plaintext_mode,
            }
            self._save_progress()
            raise


def main() -> None:
    """Main entry point for e11ocutionist."""
    try:
        # Example usage
        config = PipelineConfig(
            input_file=Path("path/to/input.xml"),
            output_dir=Path("path/to/output"),
            start_step=ProcessingStep.CHUNKING,
            force_restart=False,
            backup=False,
            chunker_model="gpt-4",
            chunker_temperature=0.2,
            entitizer_model="gpt-4",
            entitizer_temperature=0.1,
            orator_model="gpt-4",
            orator_temperature=0.7,
            tonedown_model="gpt-4",
            tonedown_temperature=0.1,
            orator_all_steps=True,
            orator_sentences=False,
            orator_words=False,
            orator_punctuation=False,
            orator_emotions=False,
            min_em_distance=None,
            dialog_mode=True,
            plaintext_mode=False,
            verbose=False,
            debug=False,
        )
        pipeline = E11ocutionistPipeline(config)
        result = pipeline.run()
        logger.info("Processing completed: %s", result)

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise


if __name__ == "__main__":
    main()
