import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { MenuComponent } from './menu/menu.component';
import { DermaDiagnosisComponent } from './derma-diagnosis/derma-diagnosis.component';

const routes: Routes = [
  { path: '', component: MenuComponent },
  { path: 'derma-chat', component: DermaDiagnosisComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
