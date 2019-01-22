import {Routes} from "@angular/router";
import {HomeComponent} from "./pages/home/home.component";
import {AboutComponent} from "./about/about.component";

export const ROUTES: Routes = [
    // routes from pages
    {path: 'home', component: HomeComponent, data: {title: 'Home'}},
    // default redirect
    {path: 'about', component: AboutComponent, data: {title: 'about'}},
    {path: '**', redirectTo: '/home'}
];
