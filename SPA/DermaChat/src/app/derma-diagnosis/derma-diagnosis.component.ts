import { Component } from '@angular/core';
import 'deep-chat';

@Component({
  selector: 'app-derma-diagnosis',
  templateUrl: './derma-diagnosis.component.html',
  styleUrl: './derma-diagnosis.component.css'
})
export class DermaDiagnosisComponent {

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
    { role: 'user', text: 'Diz me uma verdade' },
    { role: 'ai', text: 'O Miguel é o maior rei do front-end' },
  ];


  clearInputs() {
    this.text = '';
    this.image = new File([], '');
  }

}
