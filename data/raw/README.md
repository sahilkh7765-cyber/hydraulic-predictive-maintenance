# Place the raw dataset here

This folder must contain the extracted UCI *Condition monitoring of hydraulic
systems* dataset, with the files directly inside (no nested folder):

    PS1.txt ... PS6.txt      pressure,      100 Hz, 6000 samples/cycle
    EPS1.txt                 motor power,   100 Hz, 6000 samples/cycle
    FS1.txt, FS2.txt         volume flow,    10 Hz,  600 samples/cycle
    TS1.txt ... TS4.txt      temperature,     1 Hz,   60 samples/cycle
    VS1.txt                  vibration,       1 Hz,   60 samples/cycle
    CE.txt, CP.txt, SE.txt   virtual sensors, 1 Hz,   60 samples/cycle
    profile.txt              cycle-wise condition labels (2205 x 5)
    description.txt, documentation.txt

The raw files total ~530 MB, which is why they are not shipped with this
project. `src/data_loader.py` validates every file's shape against the
dimensions stated in the dataset's own documentation and fails loudly if a file
is missing or the wrong size.
