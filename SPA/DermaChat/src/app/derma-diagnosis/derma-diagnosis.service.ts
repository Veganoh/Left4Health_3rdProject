import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DermaDiagnosisService {

  private apiUrl = 'http://127.0.0.1:5000/api/diagnosis'; // URL a mudar!


  constructor(private http: HttpClient) { }

  public getDiagnosis(image: File, text: string): Observable<any> {
      return this.http.post(this.apiUrl, {image, text});
  }

}



