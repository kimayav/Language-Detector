import tkinter as tk
import customtkinter
import pickle
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
le = LabelEncoder()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("green")


class LanguagePredictorGUI:
    def _init_(self, master):
        self.master = master
        master.title("Language Predictor")

        self.label = tk.Label(master, text="Enter Text to predict Language:")
        self.label.pack()

        self.text_box = tk.Text(
            master, height=10, width=50, font=("Helvetica", 16))
        self.text_box.pack()

        self.button = customtkinter.CTkButton(
            master, text="Predict", command=self.predict_language)
        self.button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()
        progressbar = customtkinter.CTkProgressBar(master=root)
        progressbar.pack(padx=20, pady=10)
        progressbar.set(self.ac)
        self.result_label.config(text=f"Model Accuracy: {self.ac * 100}%")

    def predict_language(self):
        text = self.text_box.get("1.0", "end-1c")
        predicted_language = self.predict(text)
        self.result_label.config(
            text=f"Predicted language: {predicted_language}\nModel Accuracy: {self.ac * 100}%")

    def predict(self, text):
        model = pickle.load(open("./model.pkl", "rb"))
        cv = pickle.load(open("./transform.pkl", "rb"))

        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', '', text)
        text = re.sub(r'[[]]', '', text)
        text = text.lower()
        dat = [text]

        vect = cv.fit_transform(dat).toarray()
        my_pred = model.predict(vect)
        ac = accuracy_score(y_test, my_pred)
        self.ac = ac
        my_pred = le.inverse_transform(my_pred)
        return my_pred


root = customtkinter.CTk()
my_gui = LanguagePredictorGUI(root)
root.mainloop()