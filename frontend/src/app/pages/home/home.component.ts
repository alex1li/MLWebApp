import {Component, OnInit} from '@angular/core';
import {TweetService} from "./tweet.service";
import {
    Tweet,
    ProbabilityPrediction,
    SVCParameters,
    SVCResult
} from "./types";

@Component({
    selector: 'home',
    templateUrl: './home.component.html',
    styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

    public svcParameters: SVCParameters = new SVCParameters();
    public svcResult: SVCResult;
    public probabilityPredictions: ProbabilityPrediction[];
    public tweet: Tweet = new Tweet();

    // graph styling
    public colorScheme = {
        domain: ['#1a242c', '#e81746', '#e67303', '#f0f0f0']
    };

    constructor(private tweetService: TweetService) {
    }

    ngOnInit() {
    }

    public trainModel() {
        this.tweetService.trainModel(this.svcParameters).subscribe((svcResult) => {
            this.svcResult = svcResult;
        });
    }

    public predictIris() {
        this.tweetService.predictIris(this.tweet).subscribe((probabilityPredictions) => {
            this.probabilityPredictions = probabilityPredictions;
        });
    }

}
