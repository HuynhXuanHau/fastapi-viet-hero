from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="D:\\HOCTAP\\Hoc_ki_4\\AI\\resnet_api\\app\\models\\resnet50_final_t4_optimized.keras",  # Đường dẫn đến file model trên máy bạn
    path_in_repo="resnet50_final_t4_optimized.keras",     # Tên file trên repo
    repo_id="HXHau/fastapi-viet-hero",             # Tên repo bạn tạo trên Hugging Face
    repo_type="model"                                     # Đây là repo loại model
)