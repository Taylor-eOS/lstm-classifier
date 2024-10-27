def convert_to_mmss(timestamps):
    return ' '.join(f"{int(t)//60:02}:{int(t)%60:02}" for t in timestamps.split())

def convert_to_seconds(timestamps):
    return ' '.join(str(int(m) * 60 + int(s)) for m, s in (t.split(':') for t in timestamps.split()))

if __name__ == "__main__":
#    while True:
#        user_input = input("Timestamps: ")
#        if user_input.lower() == "exit":
#            break
#        elif ":" in user_input:
#            print("", convert_to_seconds(user_input))
#        else:
#            print("", convert_to_mmss(user_input))
    user_input = input("Timestamps: ")
    if ":" in user_input:
        print("", convert_to_seconds(user_input))
    else:
        print("", convert_to_mmss(user_input))
