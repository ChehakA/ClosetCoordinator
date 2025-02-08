import os
from typing import Dict, List, Optional, Union
import pandas as pd
from PIL import Image
from pathlib import Path

class ClosetCoordinator:
    """A class to manage clothing images and their annotations."""
    
    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    ANNOTATION_FILES = {
        'attr_cloth': ('list_attr_cloth.txt', 2),
        'bbox': ('list_bbox.txt', 5, ['image_id', 'x1', 'y1', 'x2', 'y2']),
        'attr_img': ('list_attr_img.txt', 2),
        'category_cloth': ('list_category_cloth.txt', 2, ['image_id', 'category']),
        'landmarks': ('list_landmarks.txt', 5, ['image_id', 'lm1', 'lm2', 'lm3', 'lm4'])
    }

    def __init__(self, images_base_path: Union[str, Path], annotations_dir: Union[str, Path]):
        """
        Initialize the ClosetCoordinator.

        Args:
            images_base_path: Base directory containing clothing images
            annotations_dir: Directory containing annotation files
        """
        self.images_base_path = Path(images_base_path)
        self.annotations_dir = Path(annotations_dir)
        self.validate_paths()

    def validate_paths(self) -> None:
        """Validate that the provided paths exist."""
        if not self.images_base_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_base_path}")
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")

    def build_image_lookup_table(self) -> pd.DataFrame:
        """
        Recursively scan the images directory and build a DataFrame with image information.
        
        Returns:
            DataFrame containing image metadata
        """
        image_info: List[Dict] = []
        
        for file_path in self.images_base_path.rglob("*"):
            if file_path.suffix.lower() in self.VALID_IMAGE_EXTENSIONS:
                image_info.append({
                    'image_id': file_path.name,
                    'folder': file_path.parent.name,
                    'file_name': file_path.name,
                    'file_path': str(file_path)
                })
        
        if not image_info:
            raise ValueError(f"No valid images found in {self.images_base_path}")
            
        return pd.DataFrame(image_info)

    def read_annotation_file(
        self,
        file_path: Path,
        expected_cols: int,
        skiprows: int = 0,
        header: Optional[int] = None,
        col_names: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Read an annotation file and return it as a DataFrame.

        Args:
            file_path: Path to the annotation file
            expected_cols: Minimum number of expected columns
            skiprows: Number of rows to skip at start of file
            header: Row number to use as column headers
            col_names: Optional list of column names to use

        Returns:
            DataFrame containing the annotation data, or None if reading fails
        """
        try:
            df = pd.read_csv(
                file_path,
                sep=r'\s+',
                skiprows=skiprows,
                header=header,
                engine='python',
                on_bad_lines='skip'
            )

            if df.shape[1] < expected_cols:
                print(f"Warning: {file_path.name} has {df.shape[1]} column(s); expected at least {expected_cols}")
                return None

            if col_names:
                if df.shape[1] > len(col_names):
                    extra_cols = [f'col{i}' for i in range(len(col_names), df.shape[1])]
                    col_names.extend(extra_cols)
                df.columns = col_names[:df.shape[1]]
            else:
                df.columns = ['image_id'] + [f'attr{i}' for i in range(1, df.shape[1])]

            return df

        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return None

    def load_annotations(self) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Load all annotation files from the annotations directory.

        Returns:
            Dictionary mapping annotation types to their respective DataFrames
        """
        annotations: Dict[str, Optional[pd.DataFrame]] = {}

        for anno_type, (filename, expected_cols, *extra_args) in self.ANNOTATION_FILES.items():
            file_path = self.annotations_dir / filename
            if not file_path.exists():
                print(f"Warning: Annotation file not found: {file_path}")
                annotations[anno_type] = None
                continue

            skiprows = 2 if anno_type in {'attr_cloth', 'bbox'} else 0
            col_names = extra_args[0] if extra_args else None
            
            df = self.read_annotation_file(
                file_path,
                expected_cols=expected_cols,
                skiprows=skiprows,
                col_names=col_names
            )
            annotations[anno_type] = df

        return annotations

    def merge_annotations(self, df_images: pd.DataFrame, annotations: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
        """
        Merge image lookup table with annotation DataFrames.

        Args:
            df_images: DataFrame containing image information
            annotations: Dictionary of annotation DataFrames

        Returns:
            Merged DataFrame containing all image and annotation data
        """
        merged_df = df_images.copy()
        
        for anno_type, df in annotations.items():
            if df is not None:
                try:
                    merged_df = pd.merge(merged_df, df, on='image_id', how='left')
                except Exception as e:
                    print(f"Error merging {anno_type}: {str(e)}")
            else:
                print(f"Skipping merge for {anno_type}: No data available")

        return merged_df

    def get_merged_data(self) -> pd.DataFrame:
        """
        Get complete merged dataset of images and annotations.

        Returns:
            DataFrame containing all image and annotation data
        """
        df_images = self.build_image_lookup_table()
        annotations = self.load_annotations()
        return self.merge_annotations(df_images, annotations)


def main():
    """Entry point when running the script directly."""
    try:
        images_path = Path('/Users/sophiakurz/Desktop/Category_and_Attribute_Prediction/img_backup')
        annotations_path = Path('/Users/sophiakurz/Desktop/Category_and_Attribute_Prediction/Anno_coarse')
        
        coordinator = ClosetCoordinator(images_path, annotations_path)
        merged_data = coordinator.get_merged_data()
        
        print("\nFirst few rows of merged data:")
        print(merged_data.head())
        
        print("\nDataset summary:")
        print(f"Total images: {len(merged_data)}")
        print(f"Categories: {merged_data['folder'].nunique()}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == '__main__':
    main()
