class User:
    """Which User is using these functions? Used to determine filepaths"""    
    CHOICES = ['Leonard', 'Cam-DESKTOP', "Cam-Laptop", 'Cam-Server']
    
    L = CHOICES.index('Leonard')
    CD = CHOICES.index('Cam-DESKTOP')
    CL = CHOICES.index('Cam-Laptop')
    CS = CHOICES.index('Cam-Server')

class Exercise:
    """A custom class to store exercise names and their corresponding directory names"""
    CHOICES = ['STS-5', 'Quiet-Standing-Eyes-Open', 'Quiet-Standing-Eyes-Closed', 
               'Foam-Quiet-Standing-Eyes-Open', 'Foam-Quiet-Standing-Eyes-Closed', 
               'Semi-Tandem-Balance', 'Tandem-Balance', 'Unilateral-Stance-Eyes-Open', 
               'Unilateral-Stance-Eyes-Closed', 'Get-Up-And-Go-Front-View', '3m-walk-Front-View']    

    STS5 = CHOICES.index('STS-5') #STS-5
    QSEOFS = CHOICES.index('Quiet-Standing-Eyes-Open') #Quiet standing, eyes open, firm surface
    QSECFS = CHOICES.index('Quiet-Standing-Eyes-Closed') #Quiet standing, eyes closed, firm surface
    QSEOF = CHOICES.index('Foam-Quiet-Standing-Eyes-Open') #Quiet standing, eyes open, foam
    QSECF = CHOICES.index('Foam-Quiet-Standing-Eyes-Closed') #Quiet standing, eyes closed, foam
    STS = CHOICES.index('Semi-Tandem-Balance') #Semi-tandem stance
    TS = CHOICES.index('Tandem-Balance') #Tandem stance
    USEO = CHOICES.index('Unilateral-Stance-Eyes-Open') #Unilateral stance, eyes open
    USEC = CHOICES.index('Unilateral-Stance-Eyes-Closed') #Unilateral stance, eyes closed
    TUG = CHOICES.index('Get-Up-And-Go-Front-View') #TUG
    MW = CHOICES.index('3m-walk-Front-View') #3m Walk
