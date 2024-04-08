import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DermaDiagnosisService {

  private apiUrl = 'http://127.0.0.1:5000/api/diagnosis';


  constructor(private http: HttpClient) { }

  getTextDiagnosis(text: string): Observable<any> {
    const inputData = { "User_input": text };
    return this.http.post<any>(`${this.apiUrl}/text`, inputData);
  }

  getImageDiagnosis(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('image', file);
    
    return this.http.post<any>(`${this.apiUrl}/image`, formData); 
  }

}



