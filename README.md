# sign-language-website
The AI Sign Language Interpreter website 
Dataset link: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data?select=WLASL_v0.3.json

Data preprocessing:
1.  Run download_videos.py to download videos from dataset.
2.  Run remove_non_mp4.py to remove videos that are not in mp4 format.
3.  Run remove_corrupted_mp4.py to delete corrupted videos.
4.  Run classes.py to extract classes from cleaned dataset.

To run website:
python main.py
http://localhost:3000/