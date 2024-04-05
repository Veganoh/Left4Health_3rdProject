import { Component } from '@angular/core';
import 'deep-chat';
import { RequestInterceptor } from 'deep-chat/dist/types/interceptors';

@Component({
  selector: 'app-derma-diagnosis',
  templateUrl: './derma-diagnosis.component.html',
  styleUrl: './derma-diagnosis.component.css'
})

export class DermaDiagnosisComponent {
  disease_intent = 'Melanoma'
  inputType = 'Text and Image';
  text: string = '';
  image!: File;


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
    { role: 'user', text: 'Welcome to DermaChat!' },
    { role: 'ai', text: 'O Miguel é o maior rei do front-end' },
  ];

  handleDiagnosticsClick() {
   
    // Modify the request object to include additionalBodyProps with disease_intent
    const chatElement = document.getElementById('chat-element');
    if (chatElement)
      {
        var request = JSON.parse((chatElement.getAttribute('request') || ''));
        // Add additionalBodyProps object if it doesn't exist
        request.additionalBodyProps = request.additionalBodyProps || {};

        // Add disease_intent to additionalBodyProps
        request.additionalBodyProps.disease_intent = 'Melanoma';

        chatElement.setAttribute('request', JSON.stringify(request));
    } 
}

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
