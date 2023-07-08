from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.uix.bubble import Bubble
from kivy.uix.label import Label
import shutil
import os
from transformers import BioGptTokenizer, BioGptForCausalLM
import torch
from kivy.graphics import Color, Rectangle

class MainWindow(BoxLayout):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)

        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

        self.orientation = "horizontal"

        self.file_path = ""  # Add this line to initialize the file path

        self.left_layout = BoxLayout(orientation='vertical', size_hint=(.2, 1))
        self.upload_button = Button(text="Upload", size_hint=(1, .1))
        self.upload_button.bind(on_release=self.open_filechooser)
        self.left_layout.add_widget(self.upload_button)

        self.add_widget(self.left_layout)

        self.chat_layout = BoxLayout(orientation='vertical', size_hint=(.8, 1))
        self.chat_window = ScrollView(size_hint=(1, .9))

        self.chat_label = GridLayout(cols=1, spacing=10, size_hint_y=None)
        self.chat_label.bind(minimum_height=self.chat_label.setter('height'))
        self.chat_window.add_widget(self.chat_label)

        self.chat_layout.add_widget(self.chat_window)
        
        self.input_layout = BoxLayout(size_hint=(1, .1))
        self.question_input = TextInput(hint_text='Ask your question here...', size_hint=(.8, 1), multiline=False)
        self.question_input.bind(on_text_validate=self.on_enter)
        self.input_layout.add_widget(self.question_input)

        self.submit_button = Button(text='Submit', size_hint=(.2, 1))
        self.submit_button.bind(on_release=self.on_enter)
        self.input_layout.add_widget(self.submit_button)

        self.chat_layout.add_widget(self.input_layout)

        self.add_widget(self.chat_layout)

    def generate_response(self, user_input):
        with open(self.file_path, "r") as f:  # Use self.file_path instead of hard-coded path
            content = f.read()

        print("Generating response...")

        user_input = f"Context: {content}, Question: {user_input}, Answer: "
        inputs = self.tokenizer(user_input, return_tensors="pt")

        with torch.no_grad():
            beam_output = self.model.generate(
                **inputs,
                min_length=100,
                max_length=1024,
                num_beams=5,
                early_stopping=True
            )

        response = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)

        # response = "Shri Narendra Modi was sworn-in as India's Prime Minister on 30th May 2019, marking the start of his second term in office. The first ever Prime Minister to be born after Independence, Shri Modi has previously served as the Prime Minister of India from 2014 to 2019."

        response_box = BoxLayout(orientation='vertical', size_hint_y=None)
        label = Label(text=f'Bot: {response}', color=[0,0,0, 1])
        label.text_size = (580, None)  # Adjust width for the text to wrap correctly
        label.texture_update()
        label.height = label.texture_size[1]
        response_box.add_widget(label)

        response_bubble = Bubble(orientation='horizontal')
        response_bubble.add_widget(response_box)

        bubble_container = BoxLayout(size_hint=(None, None), size=(600, '50dp'))
        bubble_container.add_widget(response_bubble)

        with response_box.canvas.before:
            Color(0, 1, 0, 1)  # Green
            rectangle = Rectangle(size=response_box.size, pos=response_box.pos)
        response_box.bind(size=lambda *args: setattr(rectangle, 'size', response_box.size),
                        pos=lambda *args: setattr(rectangle, 'pos', response_box.pos))

        self.chat_label.add_widget(bubble_container)

        return response



    def on_enter(self, instance):
        question = self.question_input.text
        if question:
            user_box = BoxLayout(orientation='vertical', size_hint_y=None)
            label = Label(text=f'User: {question}', color=[0,0,0, 1])
            label.text_size = [None, None]
            label.texture_update()
            label.height = label.texture_size[1]
            user_box.add_widget(label)

            user_bubble = Bubble(orientation='horizontal')
            user_bubble.add_widget(user_box)

            bubble_container = BoxLayout(size_hint=(None, None), size=(600, '50dp'))
            bubble_container.add_widget(user_bubble)

            with user_box.canvas.before:
                Color(0, 0, 1, 1) # Blue
                rectangle = Rectangle(size=user_box.size, pos=user_box.pos)
            user_box.bind(size=lambda *args: setattr(rectangle, 'size', user_box.size),
                        pos=lambda *args: setattr(rectangle, 'pos', user_box.pos))

            self.chat_label.add_widget(bubble_container)

            response = self.generate_response(question)
            self.question_input.text = ''
            self.chat_window.scroll_to(user_bubble)



    def open_filechooser(self, instance):
        layout = BoxLayout(orientation='vertical', size_hint=(.5, .5))
        filechooser = FileChooserIconView()
        layout.add_widget(filechooser)
        select_button = Button(text="Select", size_hint=(1, .1))
        
        
        popup = Popup(title='File Upload', content=layout)
        select_button.bind(on_release=lambda x: self.upload_file(filechooser.path, filechooser.selection[0], popup))
        layout.add_widget(select_button)
        popup.open()

    def upload_file(self, path, filename, popup):
        destination_directory = "./data"
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
        file_name = os.path.basename(filename)
        destination = os.path.join(destination_directory, file_name)
        shutil.copy(filename, destination)
        self.file_path = destination  # Update the file path
        popup.dismiss()


class MyApp(App):
    def build(self):
        return MainWindow()


if __name__ == "__main__":
    MyApp().run()
