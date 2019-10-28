# Human-Activity-Recognizor
# Run Following commands 

The following commands will Process the raw data from accelerometer and provide an model + accuracy.

python Load_Raw_Data.py Dataset1\Raw Dataset1\RawWindowed \n

python Process_Raw_Data.py Dataset1\Windowed

python Split_Data.py Dataset1\Windowed Dataset1\Data Dataset1\Data

python evaluate.py Dataset1\Data

In Lines 113 - 130 in evaluate.py, you can uncomment whichever lines to test the various models.

Suggested Models are Extra Trees, Baggning or Random Forest Classfier to provide the best result

The project is still ongoing and codes are probably subjected to changes over the next few days
