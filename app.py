import streamlit as st
import os
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd
import openai
from dotenv import load_dotenv
from ClosetCoordinator import ClosetCoordinator

@dataclass
class AppConfig:
    PAGE_TITLE: str = "Closet Coordinator: Attribute-Based Outfit Recommendation"
    LAYOUT: str = "wide"
    PAGES: list = None
    POSSIBLE_COLORS: list = None

    def __post_init__(self):
        self.PAGES = ["Home", "Fashion Items", "Outfit Recommender (Matching)"]
        self.POSSIBLE_COLORS = [
            "red", "green", "blue", "orange", "yellow", 
            "purple", "pink", "black", "white", "gray"
        ]

class FashionApp:
    COLOR_COMPLEMENTS = {
        "red": "green",
        "green": "red",
        "blue": "orange",
        "orange": "blue",
        "yellow": "purple",
        "purple": "yellow",
        "black": "white",
        "white": "black",
        "pink": "gray",
        "gray": "pink"
    }

    def __init__(self):
        self.config = AppConfig()
        self.setup_environment()
        self.setup_page_config()
        self.load_data()

    def setup_environment(self) -> None:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            st.error("OpenAI API key not found. Please check your .env file.")

    def setup_page_config(self) -> None:
        st.set_page_config(
            page_title=self.config.PAGE_TITLE,
            layout=self.config.LAYOUT
        )
        st.title("👗 Closet Coordinator: Attribute-Based Outfit Recommendation System")

    def load_data(self) -> None:
        try:
            images_path = Path("img_backup")
            if not images_path.exists():
                st.error("Images directory not found!")
                return

            coordinator = ClosetCoordinator(images_path, Path("Anno_coarse"))
            self.merged_data = coordinator.get_merged_data()

            # Check for missing 'item_type' column
            if "item_type" not in self.merged_data.columns:
                st.error("Dataset missing 'item_type' column. Ensure correct file structure.")
                self.merged_data = pd.DataFrame()  # Reset data to avoid further errors

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            self.merged_data = pd.DataFrame()

    def load_image(self, file_path: str) -> Optional[Image.Image]:
        try:
            if not os.path.exists(file_path):
                return None
            return Image.open(file_path)
        except Exception:
            return None

    def render_home_page(self) -> None:
        """Display the welcome message and app purpose."""
        st.header("Welcome to Closet Coordinator!")
        st.markdown(
            """
            👋 **Closet Coordinator: Attribute-Based Outfit Recommendation System**  
            This app helps you mix & match **fashion items** based on color, style, and pattern.  
            Browse your wardrobe, get outfit recommendations, and **never mismatch again!** 🎨✨  
            """
        )
        st.image("fashion_banner.jpg", use_column_width=True)  # Add a banner image if available

    def render_fashion_items_page(self) -> None:
        """Allow users to browse available fashion items."""
        st.header("👕 Fashion Items in Your Closet")

        if self.merged_data.empty:
            st.warning("No fashion data available.")
            return

        categories = self.merged_data["item_type"].unique().tolist()
        selected_category = st.selectbox("Select a Category", categories)

        filtered_df = self.merged_data[self.merged_data["item_type"] == selected_category]

        if filtered_df.empty:
            st.warning(f"No items found for category: {selected_category}")
            return

        for _, row in filtered_df.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                image = self.load_image(row["file_path"])
                if image:
                    st.image(image, width=100)
                else:
                    st.warning("Image not found.")
            with col2:
                st.write(f"**Color:** {row['color']}")
                st.write(f"**Style:** {row['style']}")
                if "pattern" in row:
                    st.write(f"**Pattern:** {row['pattern']}")

    def get_matching_bottom(self, selected_top: Dict[str, Any]) -> Optional[pd.Series]:
        """
        Finds a matching bottom for the selected top based on color and pattern.
        """
        if self.merged_data.empty:
            return None

        bottoms_df = self.merged_data[self.merged_data["item_type"] == "bottoms"]
        if bottoms_df.empty:
            return None

        top_color = str(selected_top.get("color", "")).strip().lower()
        if not top_color:
            return bottoms_df.sample(1).iloc[0]

        complement = self.COLOR_COMPLEMENTS.get(top_color)
        if complement:
            match_df = bottoms_df[bottoms_df["color"].str.lower() == complement]
            if not match_df.empty:
                return match_df.sample(1).iloc[0]

        match_df = bottoms_df[bottoms_df["color"].str.lower() == top_color]
        if not match_df.empty:
            return match_df.sample(1).iloc[0]

        neutral_colors = {"black", "white", "gray"}
        match_df = bottoms_df[bottoms_df["color"].str.lower().isin(neutral_colors)]
        if not match_df.empty:
            return match_df.sample(1).iloc[0]

        return bottoms_df.sample(1).iloc[0]

    def render_recommender_page(self) -> None:
        st.header("Outfit Recommender")
        if self.merged_data.empty:
            st.warning("No fashion data available.")
            return

        tops_df = self.merged_data[self.merged_data["item_type"] == "tops"]
        if tops_df.empty:
            st.warning("No tops available in the wardrobe.")
            return

        top_names = tops_df["file_name"].tolist()
        top_choice = st.selectbox("Select a Top", top_names)
        selected_top = tops_df[tops_df["file_name"] == top_choice].iloc[0]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Selected Top")
            st.write(f"Color: {selected_top['color']}")
            st.write(f"Style: {selected_top['style']}")
            if "pattern" in self.merged_data.columns:
                st.write(f"Pattern: {selected_top['pattern']}")
            top_image = self.load_image(selected_top["file_path"])
            if top_image:
                st.image(top_image, use_column_width=True)
            else:
                st.warning("Top image not found.")

        if st.button("Get Matching Outfit"):
            with st.spinner("Finding matching outfit..."):
                matching_bottom = self.get_matching_bottom(selected_top)
                if matching_bottom is not None:
                    st.subheader("Recommended Bottom")
                    st.write(f"Color: {matching_bottom['color']}")
                    st.write(f"Style: {matching_bottom['style']}")
                    if "pattern" in self.merged_data.columns:
                        st.write(f"Pattern: {matching_bottom['pattern']}")
                    bottom_image = self.load_image(matching_bottom["file_path"])
                    if bottom_image:
                        st.image(bottom_image, use_column_width=True)
                    else:
                        st.warning("Bottom image not found.")
                else:
                    st.warning("No matching bottom found.")

    def run(self) -> None:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", self.config.PAGES)

        if page == "Home":
            self.render_home_page()
        elif page == "Fashion Items":
            self.render_fashion_items_page()
        elif page == "Outfit Recommender (Matching)":
            self.render_recommender_page()

if __name__ == "__main__":
    app = FashionApp()
    app.run()
