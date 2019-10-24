# Human-Activity-Recognizor
# Run Following commands 

python Load_Raw_Data.py Dataset1\Raw Dataset1\RawWindowed
python Process_Raw_Data.py Dataset1\Windowed
python Split_Data.py Dataset1\Windowed Dataset1\Data Dataset1\Data
python evaluate.py Dataset1\Data
