import { Component } from '@angular/core';
import 'deep-chat';
import { DermaDiagnosisService } from './derma-diagnosis.service';
import { RequestInterceptor } from 'deep-chat/dist/types/interceptors';

@Component({
  selector: 'app-derma-diagnosis',
  templateUrl: './derma-diagnosis.component.html',
  styleUrl: './derma-diagnosis.component.css'
})

export class DermaDiagnosisComponent {
  disease_intent = ''
  diagnosis_received = false;
  inputType = 'Text and Image';
  text: string = '';
  image!: File;

  constructor(private dermaDiagnosisService: DermaDiagnosisService) {}

  handleDiagnosticsClick() {
    
    this.dermaDiagnosisService.getDiagnosis(this.image, this.text).subscribe((response) => {
      this.disease_intent = response.disease_intent;
      this.diagnosis_received = true;
    });
  }


  changeInputType() {
    switch (this.inputType) {
      case 'Text':
        this.inputType = 'Image';
        break;
      case 'Image':
        this.inputType = 'Text and Image';
        break;
      case 'Text and Image':
        this.inputType = 'Text';
        break;
      default:
        this.inputType = 'Text';
        break;
    }
    this.clearInputs();
  }


  initialMessages = [
    { role: 'ai', text: "Welcome to our Dermatology Assistant! 🌟 Whether you have questions about a skin condition or simply want to learn more about dermatological health, you're in the right place. You have the option to start chatting with our chatbot right away, or you can upload an image of your skin concern or describe it in text to receive a diagnosis. If you choose to upload an image or describe your condition, the chatbot will provide information specifically related to the diagnosed disease. Our chatbot is here to assist you with any skin-related inquiries. Let's get started on your journey to understanding dermatology better!" },
  ];



requestInterceptor:RequestInterceptor = (details) => {
  if (this.disease_intent)
    details.body.disease_intent = this.disease_intent
  return details
}


  clearInputs() {
    this.text = '';
    this.image = new File([], '');
  }

}
