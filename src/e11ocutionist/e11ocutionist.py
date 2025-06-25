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
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


# Define processing steps as an enum
class ProcessingStep(str, Enum):
    """Pipeline processing steps."""

    CHUNKING = "chunking"
    ENTITIZING = "entitizing"
    ORATING = "orating"
    TONING_DOWN = "toning_down"
    ELEVENLABS_CONVERSION = "elevenlabs_conversion"


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

        # Create output directory if it doesn't exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.config.output_dir / "progress.json"

        logger.info(f"Output directory: {self.config.output_dir}")

        # Initialize progress file if it doesn't exist or force_restart is True
        if not self.progress_file.exists() or self.config.force_restart:
            self.progress = {}
            self._save_progress()
        else:
            self._load_progress()

    def _load_progress(self) -> None:
        """Load progress from file."""
        if not self.progress_file:
            self.progress = {}
            return

        content = self.progress_file.read_text(encoding="utf-8").strip()
        if content:
            self.progress = json.loads(content)
        else:
            self.progress = {}

    def _save_progress(self) -> None:
        """Save progress to file."""
        if not self.progress_file:
            return

        with open(self.progress_file, "w", encoding="utf-8") as f:
            if not self.progress:
                f.write("")
            else:
                json.dump(self.progress, f, indent=2)

    def _create_backup(self, file_path: Path) -> None:
        """Create a backup of the specified file.

        Args:
            file_path: Path to the file to back up
        """
        if not self.config.backup or not file_path.exists():
            return

        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise

    def _update_progress(
        self, step: ProcessingStep, output_file: str | Path, completed: bool = True
    ) -> None:
        """Update progress for a step.

        Args:
            step: The processing step that was completed
            output_file: Path to the output file from this step
            completed: Whether the step completed successfully
        """
        step_name = step.value
        self.progress[step_name] = {
            "completed": completed,
            "output_file": str(output_file),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self._save_progress()

    def run(self) -> str:
        """Run the pipeline.

        Returns:
            Path to the final output file
        """
        try:
            self._setup_output_directory()

            # If force_restart, clear progress
            if self.config.force_restart:
                self.progress = {}
                self._save_progress()

            # Run steps based on start_step
            start_step_found = False
            for step in ProcessingStep:
                if step == self.config.start_step:
                    start_step_found = True

                if start_step_found:
                    try:
                        step_method = getattr(self, f"_run_{step.value}")
                        step_method()
                    except Exception as e:
                        logger.error(f"Error in {step.value} step: {e}")
                        self._update_progress(step, "", completed=False)
                        raise

            # Get the final output file from the last completed step
            final_output = None
            for step in reversed(list(ProcessingStep)):
                if step.value in self.progress and self.progress[step.value].get(
                    "completed", False
                ):
                    final_output = self.progress[step.value]["output_file"]
                    break

            if final_output is None:
                final_output = str(self.config.output_dir)

            logger.info("Pipeline completed successfully")
            return final_output

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _run_chunking(self) -> None:
        """Run the chunking step."""
        from .chunker import process_document

        logger.info("Running chunking step")

        input_file = self.config.input_file
        output_file = self.config.output_dir / f"{input_file.stem}_step1_chunked.xml"

        # Create backup if needed
        self._create_backup(output_file)

        # Run the chunking process
        try:
            process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                model=self.config.chunker_model,
                temperature=self.config.chunker_temperature,
                verbose=self.config.verbose,
                backup=self.config.backup,
            )

            # Update progress
            self._update_progress(ProcessingStep.CHUNKING, output_file)

            logger.info(f"Chunking completed: {output_file}")

        except Exception as e:
            logger.error(f"Error in chunking step: {e}")
            self._update_progress(ProcessingStep.CHUNKING, output_file)
            raise

    def _run_entitizing(self) -> None:
        """Run the entitizing step."""
        from .entitizer import process_document

        logger.info("Running entitizing step")

        step_name = ProcessingStep.ENTITIZING.value
        prev_step = ProcessingStep.CHUNKING.value

        # Check if previous step is completed
        if prev_step not in self.progress or not self.progress[prev_step].get(
            "completed", False
        ):
            if self.config.start_step.value == step_name:
                # If starting from this step, use input file
                input_file = self.config.input_file
            else:
                msg = (
                    f"Cannot run {step_name}: previous step ({prev_step}) not completed"
                )
                raise ValueError(msg)
        else:
            input_file = Path(self.progress[prev_step]["output_file"])

        output_file = self.config.output_dir / f"{step_name}_output.xml"

        try:
            process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                model=self.config.entitizer_model,
                temperature=self.config.entitizer_temperature,
                verbose=self.config.verbose,
            )

            self.progress[step_name] = {
                "completed": True,
                "output_file": str(output_file),
            }
            self._save_progress()

        except Exception as e:
            logger.error(f"Error in {step_name} step: {e}")
            raise

    def _run_orating(self) -> None:
        """Run the orating step."""
        from .orator import process_document

        logger.info("Running orating step")

        prev_step = ProcessingStep.ENTITIZING.value

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
            process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                model=self.config.orator_model,
                temperature=self.config.orator_temperature,
                verbose=self.config.verbose,
                backup=self.config.backup,
                steps=steps,
            )

            # Update progress
            self._update_progress(ProcessingStep.ORATING, output_file)

            logger.info(f"Orating completed: {output_file}")

        except Exception as e:
            logger.error(f"Error in orating step: {e}")
            self._update_progress(ProcessingStep.ORATING, output_file)
            raise

    def _run_toning_down(self) -> None:
        """Run the toning down step."""
        from .tonedown import process_document

        logger.info("Running toning down step")

        prev_step = ProcessingStep.ORATING.value

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

            process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                model=self.config.tonedown_model,
                temperature=self.config.tonedown_temperature,
                verbose=self.config.verbose,
                **em_args,
            )

            # Update progress
            self._update_progress(ProcessingStep.TONING_DOWN, output_file)

            logger.info(f"Toning down completed: {output_file}")

        except Exception as e:
            logger.error(f"Error in toning down step: {e}")
            self._update_progress(ProcessingStep.TONING_DOWN, output_file)
            raise

    def _run_elevenlabs_conversion(self) -> None:
        """Run the ElevenLabs conversion step."""
        from .elevenlabs_converter import process_document

        logger.info("Running ElevenLabs conversion step")

        prev_step = ProcessingStep.TONING_DOWN.value

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
            process_document(
                input_file=str(input_file),
                output_file=str(output_file),
                dialog=self.config.dialog_mode,
                plaintext=self.config.plaintext_mode,
                verbose=self.config.verbose,
            )

            # Update progress
            self._update_progress(ProcessingStep.ELEVENLABS_CONVERSION, output_file)

            logger.info(f"ElevenLabs conversion completed: {output_file}")

        except Exception as e:
            logger.error(f"Error in ElevenLabs conversion step: {e}")
            self._update_progress(ProcessingStep.ELEVENLABS_CONVERSION, output_file)
            raise
