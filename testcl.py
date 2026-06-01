import cl
def main():
    print("Opening simulator...")

    with cl.open() as neurons:
        print("Opened")
        print("Current timestamp:", neurons.timestamp())
if __name__ == "__main__":
    main()
