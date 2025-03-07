#!/usr/bin/env python3
"""
Enhanced ECFP6 Highlighter GUI Application

This application allows users to:
- Input SMILES strings individually or in batches via a file.
- Specify ECFP6 bits to highlight either manually or by uploading a text file.
- Visualize all processed molecules with highlighted bits.
- Download/save images of all highlighted molecules.
- Export a comprehensive report detailing present and missing bits for each molecule.

Dependencies:
- RDKit
- PyQt5
- matplotlib
- cairosvg

Usage:
    python ecfp6_highlighter_gui.py
"""

import sys
import os
from typing import List, Tuple, Set, Dict, Optional

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QColorDialog, QComboBox,
    QCheckBox, QSpinBox, QProgressBar, QTextBrowser, QScrollArea, QGroupBox
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFingerprintGenerator

import matplotlib.colors as mcolors
import cairosvg
import io

class MoleculeWidget(QWidget):
    """Widget to display individual molecule image and its report."""

    def __init__(self, identifier: str, smiles: str, report: str, image_data: bytes, output_format: str):
        super().__init__()
        self.identifier = identifier
        self.smiles = smiles
        self.report = report
        self.image_data = image_data
        self.output_format = output_format
        self.init_ui()

    def init_ui(self):
        """Initialize the layout and widgets for the molecule."""
        layout = QHBoxLayout()

        # Image Label
        self.image_label = QLabel()
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.display_image()

        # Report Label
        self.report_label = QTextBrowser()
        self.report_label.setFixedWidth(400)
        self.report_label.setText(self.report)

        layout.addWidget(self.image_label)
        layout.addWidget(self.report_label)

        self.setLayout(layout)

    def display_image(self):
        """Convert image data to QPixmap and display."""
        try:
            if self.output_format in ['SVG', 'Both']:
                # Extract SVG part if both formats are present
                if self.output_format == 'Both':
                    svg_end = self.image_data.find(b'</svg>')
                    svg_data = self.image_data[:svg_end+6]
                    png_data = self.image_data[svg_end+6:]
                else:
                    svg_data = self.image_data
                    png_data = None

                # Convert SVG to PNG for display
                png_converted = cairosvg.svg2png(bytestring=svg_data)
                pixmap = QPixmap()
                pixmap.loadFromData(png_converted)
            elif self.output_format == 'PNG':
                pixmap = QPixmap()
                pixmap.loadFromData(self.image_data)

            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio
            ))
        except Exception as e:
            self.image_label.setText("Image Rendering Failed.")
            print(f"Error displaying image for {self.identifier}: {e}")

class ECFP6HighlighterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced ECFP6 Highlighter")
        self.setGeometry(100, 100, 1400, 900)
        self.molecule_widgets = []
        self.init_ui()

    def init_ui(self):
        """Initialize the GUI layout and widgets."""
        main_layout = QVBoxLayout()

        # -----------------------
        # Input Section
        # -----------------------
        input_group = QGroupBox("Input Section")
        input_layout = QGridLayout()

        # SMILES Input
        smiles_label = QLabel("Single SMILES:")
        self.smiles_input = QLineEdit()
        self.smiles_input.setPlaceholderText("Enter a SMILES string here")
        input_layout.addWidget(smiles_label, 0, 0)
        input_layout.addWidget(self.smiles_input, 0, 1, 1, 3)

        # OR separator
        or_label = QLabel("OR")
        or_label.setAlignment(Qt.AlignCenter)
        input_layout.addWidget(or_label, 1, 1, 1, 2)

        # SMILES File Input
        smiles_file_label = QLabel("Batch SMILES File:")
        self.smiles_file_input = QLineEdit()
        self.smiles_file_input.setReadOnly(True)
        self.browse_smiles_file_button = QPushButton("Browse")
        self.browse_smiles_file_button.clicked.connect(self.browse_smiles_file)
        input_layout.addWidget(smiles_file_label, 2, 0)
        input_layout.addWidget(self.smiles_file_input, 2, 1, 1, 2)
        input_layout.addWidget(self.browse_smiles_file_button, 2, 3)

        # Bits to Highlight
        bits_label = QLabel("Bits to Highlight:")
        self.bits_input = QLineEdit()
        self.bits_input.setPlaceholderText("Enter bits manually, e.g., 784,668,1046")
        input_layout.addWidget(bits_label, 3, 0)
        input_layout.addWidget(self.bits_input, 3, 1, 1, 2)

        # OR separator for bits
        bits_or_label = QLabel("OR")
        bits_or_label.setAlignment(Qt.AlignCenter)
        input_layout.addWidget(bits_or_label, 4, 1, 1, 2)

        # Bits File Input
        bits_file_label = QLabel("Bits File:")
        self.bits_file_input = QLineEdit()
        self.bits_file_input.setReadOnly(True)
        self.browse_bits_file_button = QPushButton("Browse")
        self.browse_bits_file_button.clicked.connect(self.browse_bits_file)
        input_layout.addWidget(bits_file_label, 5, 0)
        input_layout.addWidget(self.bits_file_input, 5, 1, 1, 2)
        input_layout.addWidget(self.browse_bits_file_button, 5, 3)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # -----------------------
        # Fingerprint Parameters
        # -----------------------
        fp_group = QGroupBox("Fingerprint Parameters")
        fp_layout = QHBoxLayout()

        # Radius
        radius_label = QLabel("Radius:")
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(0, 10)
        self.radius_spin.setValue(3)
        fp_layout.addWidget(radius_label)
        fp_layout.addWidget(self.radius_spin)

        # Number of Bits
        nbits_label = QLabel("Number of Bits:")
        self.nbits_spin = QSpinBox()
        self.nbits_spin.setRange(128, 16384)
        self.nbits_spin.setValue(2048)
        fp_layout.addWidget(nbits_label)
        fp_layout.addWidget(self.nbits_spin)

        fp_group.setLayout(fp_layout)
        main_layout.addWidget(fp_group)

        # -----------------------
        # Visualization Options
        # -----------------------
        vis_group = QGroupBox("Visualization Options")
        vis_layout = QHBoxLayout()

        # Highlight Color
        color_label = QLabel("Highlight Color:")
        self.color_button = QPushButton()
        self.color_button.setStyleSheet("background-color: red")
        self.color_button.clicked.connect(self.choose_color)
        self.selected_color = (1.0, 0.2, 0.2)  # Default red
        vis_layout.addWidget(color_label)
        vis_layout.addWidget(self.color_button)

        # Output Format
        format_label = QLabel("Output Format:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["SVG", "PNG", "Both"])
        vis_layout.addWidget(format_label)
        vis_layout.addWidget(self.format_combo)

        # Show Atom Indices
        self.atom_idx_checkbox = QCheckBox("Show Atom Indices")
        vis_layout.addWidget(self.atom_idx_checkbox)

        # Show Bond Indices
        self.bond_idx_checkbox = QCheckBox("Show Bond Indices")
        vis_layout.addWidget(self.bond_idx_checkbox)

        # Atom Radii
        radii_label = QLabel("Atom Radii:")
        self.radii_spin = QSpinBox()
        self.radii_spin.setRange(1, 10)
        self.radii_spin.setValue(4)
        vis_layout.addWidget(radii_label)
        vis_layout.addWidget(self.radii_spin)

        vis_group.setLayout(vis_layout)
        main_layout.addWidget(vis_group)

        # -----------------------
        # Buttons
        # -----------------------
        button_group = QWidget()
        button_layout = QHBoxLayout()

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process)
        button_layout.addWidget(self.process_button)

        self.save_all_button = QPushButton("Save All Images")
        self.save_all_button.clicked.connect(self.save_all_images)
        self.save_all_button.setEnabled(False)
        button_layout.addWidget(self.save_all_button)

        self.export_report_button = QPushButton("Export Report")
        self.export_report_button.clicked.connect(self.export_report)
        self.export_report_button.setEnabled(False)
        button_layout.addWidget(self.export_report_button)

        button_group.setLayout(button_layout)
        main_layout.addWidget(button_group)

        # -----------------------
        # Visualization Area
        # -----------------------
        vis_display_group = QGroupBox("Molecule Visualizations")
        vis_display_layout = QVBoxLayout()

        # Scroll Area to hold multiple molecule widgets
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)

        vis_display_layout.addWidget(self.scroll_area)
        vis_display_group.setLayout(vis_display_layout)
        main_layout.addWidget(vis_display_group)

        # -----------------------
        # Report Section
        # -----------------------
        report_group = QGroupBox("Missing Bits Report")
        report_layout = QVBoxLayout()

        self.report_browser = QTextBrowser()
        report_layout.addWidget(self.report_browser)

        report_group.setLayout(report_layout)
        main_layout.addWidget(report_group)

        # -----------------------
        # Progress Bar
        # -----------------------
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.setLayout(main_layout)

    def browse_smiles_file(self):
        """Open a file dialog to select a SMILES file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select SMILES File",
            "",
            "Text Files (*.txt *.smi *.csv);;All Files (*)",
            options=options
        )
        if file_name:
            self.smiles_file_input.setText(file_name)

    def browse_bits_file(self):
        """Open a file dialog to select a Bits file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Bits File",
            "",
            "Text Files (*.txt *.csv);;All Files (*)",
            options=options
        )
        if file_name:
            self.bits_file_input.setText(file_name)

    def choose_color(self):
        """Open a color dialog to select highlight color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()}")
            # Convert QColor to RGB tuple
            self.selected_color = (color.redF(), color.greenF(), color.blueF())

    def parse_bits(self, bits_str: str) -> List[int]:
        """Parse the comma-separated bits string into a list of integers."""
        try:
            bits = [int(bit.strip()) for bit in bits_str.split(',') if bit.strip().isdigit()]
            if not bits:
                raise ValueError
            return bits
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Bits must be a comma-separated list of integers (e.g., '784,668,1046').")
            return []

    def parse_bits_from_file(self, file_path: str) -> List[int]:
        """Parse bits from a text file into a list of integers."""
        bits = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    # Assume bits are comma-separated or space-separated
                    parts = line.replace(',', ' ').split()
                    for part in parts:
                        if part.isdigit():
                            bits.append(int(part))
            if not bits:
                raise ValueError
            return bits
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Failed to parse bits from file: {e}")
            return []

    def load_smiles(self) -> List[Tuple[str, str]]:
        """
        Load SMILES strings from input fields.

        Returns:
            List of tuples: (identifier, SMILES)
        """
        smiles_list = []
        if self.smiles_input.text().strip():
            identifier = "Molecule_1"
            smiles = self.smiles_input.text().strip()
            smiles_list.append((identifier, smiles))
        elif self.smiles_file_input.text().strip():
            file_path = self.smiles_file_input.text().strip()
            if not os.path.isfile(file_path):
                QMessageBox.critical(self, "File Error", f"File '{file_path}' does not exist.")
                return []
            with open(file_path, 'r') as f:
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    identifier = f"Molecule_{idx}"
                    smiles_list.append((identifier, line))
        else:
            QMessageBox.critical(self, "Input Error", "Please enter a SMILES string or select a SMILES file.")
            return []
        return smiles_list

    def get_bits_to_highlight(self) -> List[int]:
        """Retrieve bits to highlight from input fields or files."""
        bits = []
        # Check if bits are entered manually
        if self.bits_input.text().strip():
            bits = self.parse_bits(self.bits_input.text().strip())
            if bits:
                return bits

        # Else, check if bits file is provided
        elif self.bits_file_input.text().strip():
            file_path = self.bits_file_input.text().strip()
            if not os.path.isfile(file_path):
                QMessageBox.critical(self, "File Error", f"Bits file '{file_path}' does not exist.")
                return []
            bits = self.parse_bits_from_file(file_path)
            if bits:
                return bits

        # If neither, show error
        QMessageBox.critical(self, "Input Error", "Please enter bits manually or upload a bits file.")
        return []

    def generate_fingerprint(self, mol: Chem.Mol, radius: int, nbits: int) -> Tuple[Chem.DataStructs.ExplicitBitVect, Dict[int, List[Tuple[int, int]]]]:
        """
        Generate a Morgan fingerprint and retrieve bit information.

        Returns:
            Tuple of fingerprint bit vector and bit_info dictionary.
        """
        # Using GetMorganFingerprintAsBitVect with bitInfo
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=bit_info)
        return fp, bit_info

    def find_highlight_atoms_bonds(self, mol: Chem.Mol, bits_to_highlight: List[int], bit_info: Dict[int, List[Tuple[int, int]]], radius: int) -> Tuple[Set[int], Set[int]]:
        """
        Determine which atoms and bonds to highlight based on the bits to highlight.

        Returns:
            Tuple of sets: (highlight_atoms, highlight_bonds)
        """
        highlight_atoms = set()
        highlight_bonds = set()

        for bit_id in bits_to_highlight:
            if bit_id not in bit_info:
                continue  # Bit not present in fingerprint

            for center_atom_idx, rad in bit_info[bit_id]:
                # Get the atom environment (bonds) of the specified radius
                env_bond_indices = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center_atom_idx)
                # Add all bonds and connected atoms
                for bidx in env_bond_indices:
                    bond = mol.GetBondWithIdx(bidx)
                    highlight_bonds.add(bidx)
                    highlight_atoms.add(bond.GetBeginAtomIdx())
                    highlight_atoms.add(bond.GetEndAtomIdx())
                # Add the center atom
                highlight_atoms.add(center_atom_idx)

        return highlight_atoms, highlight_bonds

    def highlight_molecule(
        self,
        mol: Chem.Mol,
        highlight_atoms: Set[int],
        highlight_bonds: Set[int],
        highlight_color: Tuple[float, float, float],
        atom_radii: float,
        show_atom_indices: bool,
        show_bond_indices: bool,
        output_format: str = 'SVG'
    ) -> Optional[bytes]:
        """
        Generate the molecule image with highlighted bits.

        Args:
            mol: RDKit molecule object.
            highlight_atoms: Set of atom indices to highlight.
            highlight_bonds: Set of bond indices to highlight.
            highlight_color: RGB tuple for highlight color.
            atom_radii: Radius multiplier for highlighted atoms.
            show_atom_indices: Whether to display atom indices.
            show_bond_indices: Whether to display bond indices.
            output_format: 'SVG', 'PNG', or 'Both'.

        Returns:
            Image data in bytes.
        """
        drawer_width = 400
        drawer_height = 400

        # Choose drawer based on format
        if output_format in ['SVG', 'Both']:
            drawer = rdMolDraw2D.MolDraw2DSVG(drawer_width, drawer_height)
        elif output_format == 'PNG':
            drawer = rdMolDraw2D.MolDraw2DCairo(drawer_width, drawer_height)

        options = drawer.drawOptions()
        options.highlightColor = highlight_color
        options.addAtomIndices = show_atom_indices
        options.addBondIndices = show_bond_indices

        # Prepare molecule for drawing
        AllChem.Compute2DCoords(mol)
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=list(highlight_atoms),
            highlightBonds=list(highlight_bonds),
            highlightAtomColors={idx: highlight_color for idx in highlight_atoms},
            highlightBondColors={idx: highlight_color for idx in highlight_bonds},
            highlightAtomRadii={idx: atom_radii for idx in highlight_atoms}
        )
        drawer.FinishDrawing()

        if output_format == 'SVG':
            svg = drawer.GetDrawingText()
            return svg.encode('utf-8')
        elif output_format == 'PNG':
            png = drawer.GetDrawingText()
            return png
        elif output_format == 'Both':
            svg = drawer.GetDrawingText()
            # Convert SVG to PNG using cairosvg
            try:
                png_converted = cairosvg.svg2png(bytestring=svg)
                return svg + png_converted
            except Exception as e:
                QMessageBox.critical(self, "Rendering Error", f"Failed to convert SVG to PNG: {e}")
                return None

    def process(self):
        """Process the input SMILES and highlight specified bits."""
        # Clear previous report and images
        self.report_browser.clear()
        for widget in self.molecule_widgets:
            widget.setParent(None)
        self.molecule_widgets = []

        # Get bits to highlight
        bits_to_highlight = self.get_bits_to_highlight()
        if not bits_to_highlight:
            return  # Error message already shown

        # Load SMILES
        molecules = self.load_smiles()
        if not molecules:
            return  # Error message already shown

        # Get fingerprint parameters
        radius = self.radius_spin.value()
        nbits = self.nbits_spin.value()

        # Get visualization options
        highlight_color = self.selected_color
        output_format = self.format_combo.currentText()
        show_atom_indices = self.atom_idx_checkbox.isChecked()
        show_bond_indices = self.bond_idx_checkbox.isChecked()
        atom_radii = self.radii_spin.value() / 10.0  # Scale down for better visualization

        # Initialize report
        full_report = ""

        # Update progress bar
        self.progress_bar.setMaximum(len(molecules))
        self.progress_bar.setValue(0)

        for idx, (identifier, smiles) in enumerate(molecules, 1):
            # Update progress bar
            self.progress_bar.setValue(idx)

            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                report = f"{identifier}: Invalid SMILES string.\n"
                full_report += report
                self.report_browser.append(report)
                continue

            # Kekulize and compute 2D coords
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Chem.KekulizeException:
                pass  # Proceed without kekulization
            AllChem.Compute2DCoords(mol)

            # Generate fingerprint
            fp, bit_info = self.generate_fingerprint(mol, radius, nbits)

            # Determine present and missing bits
            present_bits = set(bits_to_highlight).intersection(set(fp.GetOnBits()))
            missing_bits = set(bits_to_highlight) - present_bits

            # Find atoms and bonds to highlight
            highlight_atoms, highlight_bonds = self.find_highlight_atoms_bonds(mol, list(present_bits), bit_info, radius)

            # Generate image
            image_data = self.highlight_molecule(
                mol=mol,
                highlight_atoms=highlight_atoms,
                highlight_bonds=highlight_bonds,
                highlight_color=highlight_color,
                atom_radii=atom_radii,
                show_atom_indices=show_atom_indices,
                show_bond_indices=show_bond_indices,
                output_format=output_format
            )

            if image_data is None:
                report = f"{identifier}: Failed to generate visualization.\n"
                full_report += report
                self.report_browser.append(report)
                continue

            # Create MoleculeWidget and add to the scroll area
            molecule_widget = MoleculeWidget(
                identifier=identifier,
                smiles=smiles,
                report=f"{identifier}:\n    SMILES: {smiles}\n    Present Bits ({len(present_bits)}): {sorted(present_bits)}\n" +
                       (f"    Missing Bits ({len(missing_bits)}): {sorted(missing_bits)}\n" if missing_bits else "    All specified bits were found and highlighted.\n"),
                image_data=image_data,
                output_format=output_format
            )
            self.scroll_layout.addWidget(molecule_widget)
            self.molecule_widgets.append(molecule_widget)

            # Append to full report
            report = f"{identifier}:\n"
            report += f"    SMILES: {smiles}\n"
            report += f"    Present Bits ({len(present_bits)}): {sorted(present_bits)}\n"
            if missing_bits:
                report += f"    Missing Bits ({len(missing_bits)}): {sorted(missing_bits)}\n"
            else:
                report += f"    All specified bits were found and highlighted.\n"
            report += "\n"
            full_report += report
            self.report_browser.append(report)

        # Enable save and export buttons if at least one molecule was processed
        if self.molecule_widgets:
            self.save_all_button.setEnabled(True)
            self.export_report_button.setEnabled(True)

        # Store the report text for exporting
        self.full_report = full_report

        QMessageBox.information(self, "Processing Complete", "All molecules have been processed.")

    def save_all_images(self):
        """Save all molecule images to a selected directory."""
        if not self.molecule_widgets:
            QMessageBox.critical(self, "Save Error", "No images to save.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Images",
            "",
            options=options
        )
        if directory:
            for widget in self.molecule_widgets:
                identifier = widget.identifier
                smiles = widget.smiles
                image_data = widget.image_data
                output_format = widget.output_format

                # Define file paths
                svg_path = os.path.join(directory, f"{identifier}.svg")
                png_path = os.path.join(directory, f"{identifier}.png")

                try:
                    if output_format in ['SVG', 'Both']:
                        # Extract SVG data
                        if output_format == 'Both':
                            svg_end = image_data.find(b'</svg>')
                            svg_data = image_data[:svg_end+6]
                            # Save SVG
                            with open(svg_path, 'wb') as f_svg:
                                f_svg.write(svg_data)
                        else:
                            svg_data = image_data
                            with open(svg_path, 'wb') as f_svg:
                                f_svg.write(svg_data)

                    if output_format in ['PNG', 'Both']:
                        # Extract PNG data
                        if output_format == 'Both':
                            svg_end = image_data.find(b'</svg>')
                            png_data = image_data[svg_end+6:]
                        else:
                            png_data = image_data
                        # Save PNG
                        with open(png_path, 'wb') as f_png:
                            f_png.write(png_data)

                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Failed to save image for {identifier}: {e}")
                    continue

            QMessageBox.information(self, "Save Complete", f"All images have been saved to {directory}.")

    def export_report(self):
        """Export the missing bits report to a text file."""
        if not hasattr(self, 'full_report') or not self.full_report.strip():
            QMessageBox.critical(self, "Export Error", "No report to export.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            "missing_bits_report.txt",
            "Text Files (*.txt);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.full_report)
                QMessageBox.information(self, "Success", f"Report saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save report: {e}")

def main():
    app = QApplication(sys.argv)
    gui = ECFP6HighlighterGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
