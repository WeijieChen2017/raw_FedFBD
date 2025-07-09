results_dir = "fbd_run"

import os
import glob
import openpyxl

folder_list = glob.glob(os.path.join(results_dir, "*_tau*"))

# write the header to the excel file
# create  the xlsx file if it does not exist
if not os.path.exists('tau.xlsx'):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.cell(row=1, column=1).value = "dataset"
    sheet.cell(row=1, column=2).value = "alpha"
    sheet.cell(row=1, column=3).value = "FA"
    workbook.save('tau.xlsx')

for i_row, folder in enumerate(folder_list):
    if "tau" in folder:
        print(folder)
        folder_name = os.path.basename(folder)
        # extract the tau value from the folder name
        segments = folder_name.split("_")

        if "tau" == segments[-1]:
            alpha = 0.5
            FA = True
        else:
            alpha = float(segments[-1])
            if "FA" in folder_name:
                FA = True
            else:
                FA = False

        dataset = segments[0]

        # write the tau value to the excel file
        with openpyxl.load_workbook('tau.xlsx') as workbook:
            sheet = workbook.active
            sheet.cell(row=i_row+2, column=1).value = dataset
            sheet.cell(row=i_row+2, column=2).value = alpha
            sheet.cell(row=i_row+2, column=3).value = FA
            workbook.save('tau.xlsx')
