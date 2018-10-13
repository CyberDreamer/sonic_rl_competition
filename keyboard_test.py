import msvcrt

while True:
    key=msvcrt.getch()
    print(key)
    if key==b'r': 
        print("Wow!")