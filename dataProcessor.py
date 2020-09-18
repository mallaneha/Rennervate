import csv
from statistics import median


def save_ear(ear_list, mar_list, filename):
    # if not os.path.exists(f"{filename}.csv"):
    #     with open(f"{filename}.csv", mode="w") as train_file:
    #         file_write = csv.writer(
    #             train_file, delimiter=",", quoting=csv.QUOTE_MINIMAL
    #         )
    #         file_write.writerow(ear_list)
    #         file_write.writerow(mar_list)
    # else:
    with open(f"{filename}_mear.csv", mode="a") as file:
        file_write = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        file_write.writerow(ear_list)
        file_write.writerow(mar_list)

    # with open(f"{filename}_mar.csv", mode="a") as file:
    #     file_write = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    #     file_write.writerow(mar_list)


def processingCSV(filename):
    medMEar = []
    if filename == "Drowsy":
        status = 1
    else:
        status = 0

    with open(f"{filename}_mear.csv", mode="r") as file:
        csv_read = csv.reader(file, delimiter=',')
        count = 0

        for row in csv_read:
            # print(count)
            if row != []:
                mearList = []

                for item in row:
                    mearList.append(float(item))
                # print(mearList)
                medMEar.append(round(median(mearList), 2))
                # print(median(mearList))
                mearList.clear()
                count += 1

                if count % 2 == 0:
                    medMEar.append(status)
                    # print(medMEar)
                    finalCSV(filename, medMEar)
                    medMEar.clear()


def finalCSV(filename, medMEar):
    with open(f"{filename}_final_mear.csv", mode="a") as file:
        file_write = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        file_write.writerow(medMEar)


def main():
    processingCSV("Drowsy")
    processingCSV("NotDrowsy")


if __name__ == "__main__":
    main()
